import collections
import torch

def load_state(net, checkpoint):
    source_state = checkpoint
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)
    
def run_demo(net, data):
    net = net.eval()
    output = net(data)
    output =  torch.softmax(output, dim=1)
    return output
