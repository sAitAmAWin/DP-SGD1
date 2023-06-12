import torch
from torch import nn
from torch.utils.data import DataLoader

def compute_loss(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    print("length of dataloader")
    print(l)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss_func(log_probs, target).item()
        # get the index of the max log-probability
        # y_pred = log_probs.data.max(1, keepdim=True)[1]
    print("test_loss")
    print(test_loss)
    test_loss/=l
    print("aveg test_loss")
    print(test_loss)
    return test_loss