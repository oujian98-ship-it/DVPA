import numpy as np
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis import rdp

from src.models.model import get_model
from src.datasets.dataset_utils import AdultIncomeDataset

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR



class Client:
    def __init__(self, id, train_dataset, test_dataset, local_epochs, epsilon, delta, num_rounds, num_clients, lr, bs,
                 rate, dataset_name, cprint_rounds, device):
        '''Initialize each client according to its parameters.
            Args:
            1. epochs (int): frequency of running local training per round
            2. rounds (int): frequency of running global aggregation
            3. cprint_rounds (int): frequency of printing client information
            4. id (int): client's ID that distinguishes the certain client from the others
            5. train_dataset: client's local training dataset
            6. test_dataset: client's local test dataset
            7. device (bool): boolean that specifies if the gpu is available (True, o.w. if run on cpu -> False)
            8. local_model (Module): client's machine learning model to be locally trained
            9. num_clients (int): number of all clients in system
            10. local_noise_multiplier (float): parameter of noise in Differential Privacy (DP)
            11. local_clip_value (float): client's clipping value: clip(g,c)=1/max(1,norm(g)/c)
            12. dp_eps_budget (float): client's total privacy budget in (delta, epsilon)-DP formulation
            13. delta (float): client's delta value in (epsilon,delta)-DP, quantifies how much privacy guarantee can be exceeded
            14. alphas (list of floats): opacus default alpha choices in (alpha, epsilon)-Renyi DP (RDP) formulation
            15. best_alpha (float): client's best alpha that minimizes epsilon in (alpha, epsilon)-RDP formulation
            16. rdp_eps_spent (list of float): client's epsilons in (alpha, epsilon)-RDP formulation w.r.t. each alpha choice
            17. best_rdp_eps_spent (float): client's best accumulated epsilon spending among all (alpha, epsilon) pairs
            18. dp_eps_spent (float): client's accumulated epsilon spending in (epsilon,delta)-DP formulation
            19. num_clients: number of clients who are participating in the federated learning
            20. learning_rate (float): learning rate used in client's local training
            21. batch_size (float): batch_size used in client's local training/evaluation
            22. sample_rate (float): server applies poisson sampling with this rate to sample/not the client
            23. dataset_name (str): name of the dataset on which the client model is about to be trained

            24. loss: measure difference b/w predicted probabs. and true labels, LogSoftmax + Negative Log Likelihood
            25. optimizer: initializes SGD optimizer in PyTorch
            26. scheduler: adjusts learning rate during training for better convergence and performance
            27. rate: multiplying factor for learning rate
        '''
        self.epochs = int(local_epochs)
        self.rounds = num_rounds
        self.cprint_rounds = int(cprint_rounds)
        self.id = id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.local_model = get_model(dataset_name, bs).to(self.device)
        self.num_clients = None
        self.local_noise_multiplier = None
        self.local_clip_value = None
        self.dp_eps_budget = float(epsilon)
        self.delta = float(delta)
        self.alphas = ([1 + x / 10000000000.0 for x in range(1, 100)] + [1 + x / 10000000.0 for x in range(1, 100)] +
                       [1 + x / 10000.0 for x in range(1, 100)] + [1 + x / 10.0 for x in range(1, 9)] + [x / 2.0 for x
                                                                                                         in
                                                                                                         range(4, 300)])
        self.best_alpha = self.alphas[0]
        self.rdp_eps_spent = np.array([0.0 for _ in range(len(self.alphas))])
        self.best_rdp_eps_spent = None
        self.dp_eps_spent = 0.0
        self.previous_dp_eps_spent = 0.0
        self.num_clients = int(num_clients)
        self.learning_rate = float(lr)
        self.batch_size = int(bs)
        self.sample_rate = None
        self.dataset_name = dataset_name

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
        self.rate = rate

    def set_local_sigma_and_sample_rate(self, sample_rate):
        ''' Set the sample rate for the client and compute the local noise multiplier '''
        self.sample_rate = float(sample_rate)
        self.local_noise_multiplier = get_noise_multiplier(
            target_epsilon=self.dp_eps_budget,
            target_delta=self.delta,
            sample_rate=self.sample_rate,
            epochs=self.rounds)
        return self.local_noise_multiplier

    def set_local_clip_value(self, global_clip_value, global_sigma):
        ''' Compute and return the local clip value of the client, given the global parameters -
        global sigma and global clip value '''
        self.local_clip_value = global_clip_value * (global_sigma / self.local_noise_multiplier)
        return self.local_clip_value

    def train(self, global_model, current_round):
        '''
        Inputs:
        1. global_model (Module): updated global model,
        2. current_round (int): number of rounds running so far.

        Return:
        1. {k: v * lr ...} (dictionary): trained local model, to be shared with server for aggregation.
        2. dp_eps_spent (float): accumulated privacy spent up to next round, to be shared with server for saving in file.
        3. local_clip_value (float): clipping value used by client
        4. net_norm (float): (network-wise) norm of updates / sampling rate
        5. (train_losses, train_num)
        6. (test_correct, test_num)

        Training function key steps:
        1. Receive global model from server.
        2. Train the given model locally according to the given hyperparameters.
        3. Run self.clip_layer_parameters (or self.clip_parameters) to clip the model updates layer wise (network wise).
        4. Run self.add_noise to add local noise.
        5. Run self.rdp_accountant to measure how much privacy budget is spent (use 1.0 instead of sample_rate).

        Note: Comment out Step 4 if want to use baseline training with fixed clipping value over time.
        '''
        self.local_model.load_state_dict(global_model.state_dict())
        self.local_model.to(self.device)
        test_global_model_local_data, test_num_global_model_local_data = self.test_metrics()

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.local_model.train()
        initial_params = {k: v.clone() for k, v in self.local_model.state_dict().items()}

        if current_round == 0:
            unique_layers = set(map(lambda name: name.split('.')[0], initial_params.keys()))
            self.num_layers = len(unique_layers)

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = self.loss(outputs, labels)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()  
            self.scheduler.step()

        test_local_model_local_data, test_num_local_model_local_data = self.test_metrics()
        train_losses, train_num = self.train_metrics()
        test_local_model_global_data = self.evaluate()
        updates = {k: (self.local_model.state_dict()[k] - v) / (self.learning_rate * self.rate) for k, v in
                   initial_params.items()}

        net_norm = self.calculate_parameters_norm(updates)

        self.clip_layer_parameters(updates, self.num_layers)
        net_norm2 = self.calculate_parameters_norm(updates)

        self.add_noise(updates)
        net_norm3 = self.calculate_parameters_norm(updates)

        self.rdp_accountant()
        if current_round % self.cprint_rounds == 0:
            print(f"Rnd: {current_round}, CID: {self.id}, "
                  f"Norm1: {net_norm:.2f}, Norm2: {net_norm2:.2f}, Norm3: {net_norm3:.2f}, "
                  f"Clip: {self.local_clip_value:.2f}, "
                  f"Srate: {self.sample_rate}, Sigma: {self.local_noise_multiplier:.2f}, "
                  f"REeps: {self.dp_eps_budget - self.dp_eps_spent:.2f}")

        self.per_round_dp_eps_spent = self.dp_eps_spent - self.previous_dp_eps_spent

        return {k: v * self.learning_rate * self.rate for k, v in updates.items()}, \
            self.dp_eps_spent, \
            self.per_round_dp_eps_spent, \
            net_norm, \
            train_losses, \
            train_num, \
            test_local_model_local_data, \
            test_num_local_model_local_data, \
            test_global_model_local_data, \
            test_num_global_model_local_data, \
            test_local_model_global_data, \
            self.local_noise_multiplier, \
            self.rdp_eps_sample, \
            self.best_alpha

    def calculate_rdp_eps_spent(self, srate):
        '''Computes RDP guarantees of the Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.'''
        noise_mult = self.local_noise_multiplier  
        noise_mult = noise_mult.cpu().numpy() if torch.is_tensor(noise_mult) else noise_mult
        srate = srate.cpu().numpy() if torch.is_tensor(srate) else srate

        rdp_eps = rdp.compute_rdp(q=srate, noise_multiplier=noise_mult, steps=1.0, orders=self.alphas)
        return rdp_eps

    def calculate_best_rdp_eps_spent(self, rdp_eps):
        '''Computes epsilon given a list of RDP values at multiple RDP orders and target `delta`.
        The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
        is based on the theorem presented in the following work:
        Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
        International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
        Particularly, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.'''

        eps, best_alpha = rdp.get_privacy_spent(
            orders=self.alphas,
            rdp=copy.deepcopy(rdp_eps),
            delta=self.delta
        )

        best_rdp_eps = self.rdp_eps_spent[self.alphas.index(best_alpha)]
        return eps, best_alpha, best_rdp_eps

    def rdp_accountant(self):
        '''
        Inputs:
        1. sample_rate (float): sampling rate (client sets it to 1.0, but server motivates client with lower value.)

        Accountant method:
        1. Accumulate the client's RDP epsilon spending over rounds w.r.t. sampling_rate and all choices of alphas.
        2. Find the best (alpha, epsilon)-RDP guarantee and update the accumulated epsilon of DP guarantee per round.
        '''
        new_rdp_eps_spent = np.array(self.calculate_rdp_eps_spent(self.sample_rate))
        self.rdp_eps_sample = new_rdp_eps_spent[0]
        self.rdp_eps_spent = (np.array(self.rdp_eps_spent) + new_rdp_eps_spent).tolist()
        eps, best_alpha, best_rdp_eps = self.calculate_best_rdp_eps_spent(self.rdp_eps_spent)
        self.best_rdp_eps_spent = best_rdp_eps
        self.best_alpha = best_alpha
        self.previous_dp_eps_spent = copy.copy(self.dp_eps_spent)
        self.dp_eps_spent = eps

    def calculate_parameters_norm(self, parameters):
        '''Calculate network-wise Frobenius norm, considering network as all layers together.'''
        layer_inner_products = [pow(torch.linalg.norm(layer_param), 2) for _, layer_param in parameters.items()]
        return pow(sum(layer_inner_products), 0.5)

    def clip_parameters(self, parameters):
        ''' Performs flat network-wise clipping, i.e., layer params * min(1, C/(||network params||_2 + small No.)).'''
        network_frobenius_norm = self.calculate_parameters_norm(parameters)
        if network_frobenius_norm > self.local_clip_value:
            clip_scalar = min(1.0, self.local_clip_value / (network_frobenius_norm + 1e-6))
            for _, layer_param in parameters.items():
                layer_param.mul_(clip_scalar)

    def clip_layer_parameters(self, parameters, num_layers):
        '''
        Performs flat layer-wise clipping, i.e. layer params * min(1, C/(||layer params||_2 + small No.)).
        '''
        for layer_name, layer_param in parameters.items():
            param = {layer_name: layer_param}
            layer_frobenius_norm = self.calculate_parameters_norm(param)
            layer_clip_value = self.local_clip_value / np.sqrt(num_layers)
            if layer_frobenius_norm > layer_clip_value:
                clip_scalar = min(1.0, layer_clip_value / (layer_frobenius_norm + 1e-6))
                layer_param.mul_(clip_scalar)

    def add_noise(self, parameters):
        ''' Add Gaussian noise: noise_multiplier * N(0,I), layer-wise.'''
        for _, layer_param in parameters.items():
            local_noise_std = self.local_noise_multiplier * self.local_clip_value / (np.sqrt(self.num_clients))
            noise = torch.normal(0, local_noise_std, size=layer_param.shape).to(self.device)
            layer_param.add_(noise)

    def test_metrics(self):
        ''' Return the number of correct predictions and the total number of test samples '''
        testloaderfull = DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True)

        self.local_model.eval()

        correct = 0
        test_num = 0

        with torch.no_grad():
            for images, labels in testloaderfull:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.local_model(images)
                predicted = torch.argmax(output, dim=1)
                correct += (torch.sum(predicted == labels)).item()
                test_num += labels.shape[0]

        return correct, test_num

    def train_metrics(self):
        ''' Return the accumulated loss and the total number of training samples '''
        trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
        self.local_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.local_model(images)
                loss = self.loss(output, labels)
                train_num += labels.shape[0]
                losses += loss.item() * labels.shape[0]

        return losses, train_num

    def evaluate(self):
        ''' Computes the overall accuracy of the server on the test dataset '''
        if self.dataset_name == 'fmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            global_testset = torchvision.datasets.FashionMNIST(root="./data/fmnist/rawdata", train=False, download=True,
                                                               transform=transform)
            test_dataloader = torch.utils.data.DataLoader(global_testset, batch_size=self.batch_size, drop_last=False,
                                                          shuffle=True)
        elif self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            global_testset = torchvision.datasets.MNIST(root="./data/mnist/rawdata", train=False, download=True,
                                                        transform=transform)
            test_dataloader = torch.utils.data.DataLoader(global_testset, batch_size=self.batch_size, drop_last=False,
                                                          shuffle=True)
        elif self.dataset_name == 'adult_income':
            global_testset = AdultIncomeDataset(csv_file="./data/adult_income/test/adult.test")
            test_dataloader = DataLoader(global_testset, batch_size=self.batch_size, drop_last=False, shuffle=True)

        self.local_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.local_model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy