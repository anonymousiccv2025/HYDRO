from HYDRO.model.adaface import net
import torch

# From https://github.com/mk-minchul/AdaFace

def load_pretrained_model(checkpoint_path="../../model/adaface/adaface_ir50_ms1mv2.ckpt", model_name='ir_50'):
    # load model and pretrained statedict
    model = net.build_model(model_name)
    statedict = torch.load(checkpoint_path)['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model
