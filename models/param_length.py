import torch

def param_length(model):
    param = list(model.parameters())
    nc = len(param)
    length = 0
    for i in range(nc):
        param_shape = param[i].shape
        param[i].data = torch.flatten(param[i])
        length += len(param[i].data)  # 模型参数长度
        # print(length)  # 打印模型长度
    return length
