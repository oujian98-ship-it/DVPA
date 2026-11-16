import numpy as np
import torch
from src.models.model import get_model
from torchvision import transforms
from src.datasets.dataset_utils import AdultIncomeDataset
import torchvision
from collections import Counter


class Server:
    def __init__(self, clients, num_clients, rounds, sample_rate, clip_value, delta_value, lr, bs, rate,
                 current_pass, dataset_name, sprint_rounds, privacy_states, saving_sample_rates, device):
        '''Initialize server according to its parameters.
            Args:
            1. global_model (Module): server's machine learning model, updated after every global epoch
            2. clients (list): list of clients
            3. num_clients (int): number of clients which can participate in the federated learning.
            4. global_rounds (int): frequency of running global aggregation
            5. learning_rate (float): learning rate used by client's local training
            6. batch_size (int): number of client's samples processed in each training iteration
            7. sprint_rounds (int): prints the server information after each training round

            8. sample_rates (list): list of clients' sampling rates
            9. sampled_clients (list): a list of id of clients that are sampled
            10. delta_value (float): delta parameter in (epsilon,delta)-DP formulation
            11. global_sigma (float): noise multiplier set by the server globally
            12. local_sigmas (list): list of local noise multiplier of clients
            13. global_clip_value (float): clip value set by the server globally
            14. local_clip_values (list): list of clip values for every client

            15. client_updates (list): list of updates shared by the clients to server
            16. privacy_spent (list): list of privacy spent by every client
            17. net_norms (list): list of clients' updates
            18. clients_train_loss (list): list of clients' local train losses at current round
            19. clients_train_num
            20. clients_test_acc (list): list of clients' local test accuracy at current round
            21. clients_test_nums (list)
            22. clients_test_acc_local_model_local_data (list): list of clients' test accuracies on local model
            23. clients_test_num_local_model_local_data (list)
            24. clients_test_acc_global_model_local_data (list): list of clients' test accuracies on global model
            25. clients_test_num_global_model_local_data (list
            26. clients_test_acc_local_model_global_data (list): list of clients' test accuracies on global test dataset

            27. aggregated_updates_dict (dictionary): aggregated clients' updates, summed by noise if client is not sampled

            28. device (bool): boolean that specifies if the gpu is available (True, o.w. if run on cpu -> False)
            29. config_path (string): path to the configuration file for the project

            30. privacy_spent_file: path to the files that save privacy spent results
            31. accuracy_file: path to the files that save privacy spent results
            
            32. plot_clients_norm (list): list of average clients' network-wise Frobenius norm of updates, to be plotted
            33. plot_clients_train_loss (list): list of average clients' training losses, to be plotted
            34. plot_test_acc (list) : list of test accuracy on global test data, to be plotted
            35. plot_clients_test_acc_local_model_local_data (list): list of average clients' test accuracy on local model
            36. plot_clients_test_acc_global_model_local_data (list): list of average clients' test accuracy on global model
            37. plot_clients_test_acc_local_model_global_data (list): list of average clients' test accuracy on server's global test dataset
            38. plot_global_clip_values (list): list of global clip values
            39. plot_local_clip_values (list): list of average local clip values
            40. plot_clients_privacy_spent (multi-dim list): num_clients * num_rounds dimension, total privacy budget spent till a certain round 
            41. plot_clients_per_round_privacy_spent (list): privacy budget spent in every round
            42. plot_clients_per_round_noise_multiplier (list): noise multiplier in every round
            43. plot_clients_rdp_eps_sample (list): rdp epsilon of clients
            44. plot_clients_best_alpha (list): list of best alpha values             
            
            45. rate (float): multiplying factor of learning rate
            46. current_pass (int): the nth number of run for the federated learning setup
            47. dataset_name (str): name of the training dataset

            48. testset: central test dataset
            49. saving_sample_rates (float): sampling rate when privacy budget is saved by the client
            50. privacy_states: state of privacy budget saving (save) or spending (spend)
        '''
        self.global_model = get_model(dataset_name, bs)
        self.clients = clients
        self.num_clients = int(num_clients)
        self.global_rounds = int(rounds)
        self.sprint_rounds = int(sprint_rounds)
        self.learning_rate = float(lr)
        self.batch_size = int(bs)

        self.global_sample_rate = float(sample_rate)

        self.sample_rates = None
        self.sampled_clients = None
        self.delta_value = float(delta_value)
        self.global_sigma = None
        self.local_sigmas = [None for _ in range(self.num_clients)]
        self.global_clip_value = float(clip_value)
        self.local_clip_values = [0.0] * self.num_clients

        self.clients_update = [None for _ in range(self.num_clients)]
        self.clients_privacy_spent = [None for _ in range(self.num_clients)]
        self.clients_norm = [None for _ in range(self.num_clients)]
        self.clients_train_loss = [None for _ in range(self.num_clients)]
        self.clients_train_num = [None for _ in range(self.num_clients)]
        self.clients_test_acc_local_model_local_data = [None for _ in range(self.num_clients)]
        self.clients_test_num_local_model_local_data = [None for _ in range(self.num_clients)]
        self.clients_test_acc_global_model_local_data = [None for _ in range(self.num_clients)]
        self.clients_test_num_global_model_local_data = [None for _ in range(self.num_clients)]
        self.clients_test_acc_local_model_global_data = [None for _ in range(self.num_clients)]

        self.aggregated_updates_dict = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = 'config.yaml'

        self.file_clients_privacy_spent = None
        self.file_test_acc = None

        self.plot_clients_norm = []
        self.plot_clients_train_loss = []
        self.plot_test_acc = []
        self.plot_clients_test_acc_local_model_local_data = []
        self.plot_clients_test_acc_global_model_local_data = []
        self.plot_clients_test_acc_local_model_global_data = []
        self.plot_global_clip_values = [self.global_clip_value for _ in range(self.global_rounds)]
        self.plot_local_clip_values = []
        self.plot_local_sigmas = []
        self.plot_clients_privacy_spent = [[None for _ in range(self.global_rounds)] for _ in range(self.num_clients)]
        self.plot_clients_per_round_privacy_spent = [[None for _ in range(self.global_rounds)]
                                                     for _ in range(self.num_clients)]
        self.plot_clients_per_round_noise_multiplier = [[None for _ in range(self.global_rounds)]
                                                     for _ in range(self.num_clients)]
        self.plot_clients_per_round_rdp_privacy_spent = [[None for _ in range(self.global_rounds)]
                                                        for _ in range(self.num_clients)]
        self.plot_clients_per_round_best_alpha = [[None for _ in range(self.global_rounds)]
                                                        for _ in range(self.num_clients)]
        self.rate = rate

        self.current_pass = current_pass
        self.dataset_name = dataset_name
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        if dataset_name == 'fmnist':
            self.testset = torchvision.datasets.FashionMNIST(
                root="./data/fmnist/rawdata",
                train=False,
                download=True,
                transform=transform)
        elif dataset_name == 'mnist':
            self.testset = torchvision.datasets.MNIST(
                root="./data/mnist/rawdata",
                train=False,
                download=True,
                transform=transform)
        elif dataset_name == 'adult_income':
            self.testset = AdultIncomeDataset(csv_file="./data/adult_income/test/adult.test")

        self.saving_sample_rates = saving_sample_rates
        self.privacy_states = privacy_states

    def update_sample_rate_local_accounting(self, current_round):
        for client_id, client in enumerate(self.clients):
            if self.privacy_states[client_id][current_round] == 'save':
                self.sample_rates[client_id] = self.saving_sample_rates[client_id]
            privacy_spent, per_round_dp_eps_spent, per_round_rdp_eps_spent, best_alpha = client.rdp_accountant(sample_rate=self.sample_rates[client_id])
            self.clients_privacy_spent[client_id] = float(privacy_spent)
            self.plot_clients_privacy_spent[client_id][current_round] = float(privacy_spent)
            self.plot_clients_per_round_privacy_spent[client_id][current_round] = float(per_round_dp_eps_spent)
            self.plot_clients_per_round_rdp_privacy_spent[client_id][current_round] = float(per_round_rdp_eps_spent)
            self.plot_clients_per_round_best_alpha[client_id][current_round] = float(best_alpha)

    def compute_global_sigma(self):
        '''Compute global sigma by calculating the inverse of the weighted
        average over all 1/local_sigma. Reference - Section 3.4 of IDP SGD paper
        '''
        value_counts = Counter(self.local_sigmas)
        unique_values = np.array(list(value_counts.keys()))
        occurrences = np.array(list(value_counts.values()))
        summation = np.sum(occurrences / (self.num_clients * unique_values))
        global_sigma = 1 / summation
        return global_sigma

    def receive_local_sigmas(self, current_round):
        ''' Compute local sigma value for every client '''
        for client_id, client in enumerate(self.clients):
            local_sigma = client.set_local_sigma_and_sample_rate(self.global_sample_rate, current_round)
            self.local_sigmas[client_id]=local_sigma
        self.plot_local_sigmas.append(sum(self.local_sigmas) / self.num_clients)


    def send_global_privacy_parameters(self):
        ''' Compute local clip value on the client side by sharing global parameters (global clip value and
                global sigma) with the client '''
        for client_id, client in enumerate(self.clients):
            self.local_clip_values[client_id] = client.set_local_clip_value(self.global_clip_value, self.global_sigma)
        self.plot_local_clip_values.append(sum(self.local_clip_values) / self.num_clients)

    def federated_learning(self):
        ''' Perform federated learning by
        initializing privacy parameters
        invoking client training,
        adding noise for unsampled clients,
        aggregating clients updates,
        tracking privacy budgets and performance and saving in files and plotting them.
        '''
        for rnd in range(self.global_rounds):
            self.sample_rates = [self.global_sample_rate for _ in range(self.num_clients)]
            self.receive_local_sigmas(rnd)
            self.global_sigma = self.compute_global_sigma()
            self.send_global_privacy_parameters()

            self.client_operations(rnd)
            self.update_sample_rate_local_accounting(rnd)

            self.sampled_clients = self.sample_clients()
            if len(self.sampled_clients) == 0:
                continue

            self.aggregate_sampled_clients()
            self.update_global_model()

            acc_global_model_global_data = self.evaluate()
            if rnd % self.sprint_rounds == 0:
                avg_sample_rate = sum(self.sample_rates) / self.num_clients
                print(
                    f"Run: {self.current_pass}, Rnd: {rnd}, Avg Sample: {avg_sample_rate:.2f}, "
                    f"Global Clip: {self.global_clip_value:.2f}, Global Sigma: {self.global_sigma:.2f}, "
                    f"Global Test Acc: {acc_global_model_global_data:.2f}%, "
                    f"Avg Test Acc LMLD: {self.plot_clients_test_acc_local_model_local_data[-1]:.2f}%, "
                    f"Avg Test Acc GMLD: {self.plot_clients_test_acc_global_model_local_data[-1]:.2f}%, "
                    f"Avg Test Acc LMGD: {self.plot_clients_test_acc_local_model_global_data[-1]:.2f}%, "
                    f"Avg Train Loss: {self.plot_clients_train_loss[-1]:.2f}")

        torch.save(self.global_model.state_dict(), "global_model.pth")
        print(f'final accuracy: {acc_global_model_global_data}')
        return (self.plot_clients_test_acc_local_model_local_data, \
            self.plot_clients_test_acc_global_model_local_data, \
            self.plot_clients_test_acc_local_model_global_data, \
            self.plot_clients_norm, \
            self.plot_clients_train_loss, self.plot_test_acc, \
            self.plot_local_clip_values, self.plot_global_clip_values, \
            self.plot_clients_privacy_spent, self.plot_clients_per_round_privacy_spent, \
            self.plot_clients_per_round_noise_multiplier, self.plot_clients_per_round_rdp_privacy_spent, self.plot_clients_per_round_best_alpha)

    def sample_clients(self):
        ''' Performs client sampling based on client sampling rates '''
        sampled_clients = []

        for client_id, rate in enumerate(self.sample_rates):
            rate_cpu = rate.cpu().numpy() if isinstance(rate, torch.Tensor) else rate
            if np.random.choice([True, False], p=[rate_cpu, 1 - rate_cpu]):
                sampled_clients.append(client_id)
        return sampled_clients

    def client_operations(self, rnd):
        ''' Perform local training on clients, and obtain
        (1) client model updates,
        (2) privacy budget spent,
        (3) clients' test accuracy,
        (4) clients' training loss.

        Inputs:
        rnd (int): the current global round
        '''

        for client_id, client in enumerate(self.clients):
            self.global_model = self.global_model.to(self.device)

            (model_updates, net_norm, train_loss, train_num,
             client_test_acc_local_model_local_data, client_test_num_local_model_local_data,
             client_test_acc_global_model_local_data, client_test_num_global_model_local_data,
             client_test_acc_local_model_global_data, per_round_local_noise_multiplier) = \
                client.train(self.global_model, rnd)

            self.clients_update[client_id] = {k: v.to(self.device) for k, v in model_updates.items()}
            self.plot_clients_per_round_noise_multiplier[client_id][rnd] = float(per_round_local_noise_multiplier)
            self.clients_norm[client_id] = net_norm
            self.clients_train_loss[client_id] = train_loss
            self.clients_test_acc_local_model_local_data[client_id] = client_test_acc_local_model_local_data
            self.clients_test_acc_global_model_local_data[client_id] = client_test_acc_global_model_local_data
            self.clients_test_acc_local_model_global_data[client_id] = client_test_acc_local_model_global_data
            self.clients_train_num[client_id] = train_num
            self.clients_test_num_local_model_local_data[client_id] = client_test_num_local_model_local_data
            self.clients_test_num_global_model_local_data[client_id] = client_test_num_global_model_local_data

    def aggregate_sampled_clients(self):
        num_sampled_clients = len(self.sampled_clients)
        tot_net_norm = 0.0
        tot_client_train_loss = 0.0
        tot_clients_test_acc_local_model_local_data = 0.0
        tot_clients_test_acc_global_model_local_data = 0.0
        tot_clients_test_acc_local_model_global_data = 0.0
        tot_clients_train_num = 0.0
        tot_clients_test_num_local_model_local_data = 0.0
        tot_clients_test_num_global_model_local_data = 0.0
        for client_id in self.sampled_clients:
            tot_net_norm += self.clients_norm[client_id]
            tot_client_train_loss += self.clients_train_loss[client_id]
            tot_clients_test_acc_local_model_local_data += self.clients_test_acc_local_model_local_data[client_id]
            tot_clients_test_acc_global_model_local_data += self.clients_test_acc_global_model_local_data[client_id]
            tot_clients_test_acc_local_model_global_data += self.clients_test_acc_local_model_global_data[client_id]
            tot_clients_train_num += self.clients_train_num[client_id]
            tot_clients_test_num_local_model_local_data += self.clients_test_num_local_model_local_data[client_id]
            tot_clients_test_num_global_model_local_data += self.clients_test_num_global_model_local_data[client_id]

        self.plot_clients_norm.append(tot_net_norm / num_sampled_clients)
        self.plot_clients_train_loss.append(tot_client_train_loss / tot_clients_train_num)
        self.plot_clients_test_acc_local_model_local_data.append(100 * tot_clients_test_acc_local_model_local_data /
                                                                 tot_clients_test_num_local_model_local_data)
        self.plot_clients_test_acc_global_model_local_data.append(100 * tot_clients_test_acc_global_model_local_data /
                                                                  tot_clients_test_num_global_model_local_data)
        self.plot_clients_test_acc_local_model_global_data.append(tot_clients_test_acc_local_model_global_data /
                                                                  num_sampled_clients)

        self.aggregated_updates_dict = {k: torch.zeros_like(v).to(self.device) for k, v in self.clients_update[0].items()}
        averaging_coefficient = sum(self.sample_rates)
        for client_id, client_update in enumerate(self.clients_update):
            if client_id in self.sampled_clients:
                for k in self.aggregated_updates_dict.keys():
                    self.aggregated_updates_dict[k] += client_update[k] / averaging_coefficient
            else:
                for k in self.aggregated_updates_dict.keys():
                    noise_std = self.global_sigma * self.global_clip_value / np.sqrt(self.num_clients)
                    noise = torch.normal(0, noise_std, size=self.aggregated_updates_dict[k].shape).to(self.device)
                    self.aggregated_updates_dict[k] += noise * self.learning_rate * self.rate / averaging_coefficient

    def update_global_model(self):
        global_state_dict = self.global_model.state_dict()
        for k in global_state_dict.keys():
            global_state_dict[k] = global_state_dict[k].to(self.device) + self.aggregated_updates_dict[k].to(
                self.device)
        self.global_model.load_state_dict(global_state_dict)

    def evaluate(self):
        ''' Computes the overall accuracy of the server on the test dataset '''
        self.global_model.eval()
        test_dataloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, drop_last=False, shuffle=True)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        self.plot_test_acc.append(accuracy)
        return accuracy




