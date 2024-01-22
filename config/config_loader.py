mandatory = {}
class DefaultSetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __setitem__(self, key, value):
        if key not in self.dictionary:
            if value is mandatory:
                raise ValueError(f" Argument --> {key} was mandatory but is not there")
            else:
                self.dictionary[key] = value

    def __getitem__(self, key):
        if key not in self.dictionary:
            self.dictionary[key] = {}
        return DefaultSetter(self.dictionary[key])

def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.model
    config["model_type"] = args.model_type
    config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["fnet_norm"] = "group"
    config["cnet_norm"] = "group"
    config["num_scales"] = args.num_scales

    return config
