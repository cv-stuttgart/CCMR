import json
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
def load_json_config(config_path):

    file = open(config_path) 
    config = json.load(file)

    default_setter = DefaultSetter(config)
    default_setter["name"] = mandatory
    default_setter["model_type"] = "CCMR+"
    default_setter["train"]["lr"] = mandatory
    default_setter["train"]["dataset"] = mandatory
    default_setter["train"]["num_steps"] = mandatory
    default_setter["train"]["batch_size"] = mandatory
    default_setter["train"]["image_size"] = mandatory
    default_setter["train"]["validation"] = mandatory
    default_setter["train"]["restore_ckpt"] = None
    default_setter["train"]["iters"] = [4, 6, 8]
    default_setter["train"]["eval_iters"] = mandatory 
    default_setter["train"]["loss"] = mandatory
    default_setter["train"]["gamma"] = mandatory
    default_setter["train"]["wdecay"] = mandatory
    default_setter["lr_peak"] = 0.05
    default_setter["mixed_precision"] = False
    default_setter["gpus"] = [0, 1, 2]
    default_setter["epsilon"] = 1e-8
    default_setter["add_noise"] = False
    default_setter["clip"] = 1.0
    default_setter["dropout"] = 0.0
    default_setter["current_phase"] = 0
    default_setter["current_steps"] = -1
    default_setter["fnet_norm"] = "group"
    default_setter["cnet_norm"] = "group"
    default_setter["grad_acc"] = [1, 1, 1, 1]
    default_setter["cuda_corr"] = False

    default_setter["agg_num_heads"] = 8
    default_setter["agg_depth"] = 1
    default_setter["agg_mlp_ratio"] = 1
    default_setter["agg_cnet_depth"] = 1
    default_setter["agg_cnet_num_heads"] = 8
    default_setter["agg_cnet_mlp_ratio"] = 1
    default_setter["agg_cnet_shared"] = False
    default_setter["agg_shared"] = False

    default_setter["f_no_act"] = True
    default_setter["c_no_act"] = "M"
    default_setter["use_pos"] = True

    return config

def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.model
    config["model_type"] = args.model_type
    config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["fnet_norm"] = "group"
    config["cnet_norm"] = "group"
    config["cuda_corr"] = args.cuda_corr

    return config
