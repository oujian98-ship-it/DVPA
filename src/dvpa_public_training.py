"""Route pinned training-public protocols without changing legacy artifacts."""
def module(base):
    count=base.get('public_train_count',1000)
    if type(count) is not int or count not in (64,1000):
        raise ValueError('public_train_count must be 64 or 1000')
    if count==64:
        from . import dvpa_public64 as selected
    else:
        from . import dvpa_train_public as selected
    return selected

def validate_public_train_config(base):return module(base).validate_public_train_config(base)
def remove_public_training_examples(base,train):return module(base).remove_public_training_examples(base,train)
def partition_record(base):return module(base).partition_record(base)
def evaluation_record(base,sha):return module(base).evaluation_record(base,sha)
def PublicCurvatureProxy(config):return module(config['base']).PublicCurvatureProxy(config)
def verify_validation(config,proof=None):return module(config['base']).verify_validation(config,proof)
