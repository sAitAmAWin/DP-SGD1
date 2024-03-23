import matplotlib.pyplot as plt
import torch
from utils.options import args_parser

def plot_acc(args) :
    dataset = "mnist"
    model = "mlp"
    enc1 = torch.load('./save/train_data/algorithm_quan_dataset_{}_model_{}_loss.txt'.format(dataset,model))
    enc2 = torch.load('./save/train_data/algorithm_spar_dataset_{}_model_{}_loss.txt'.format(dataset,model))
    enc3 = torch.load('./save/train_data/algorithm_adaptiveQSGD_dataset_{}_model_{}_loss.txt'.format(dataset,model))
    enc4 = torch.load('./save/train_data/algorithm_SGD_dataset_{}_model_{}_loss.txt'.format(dataset,model))
    temp1 = list(enc1)
    temp2 = list(enc2)
    temp3 = list(enc3)
    temp4 = list(enc4)
    plt.plot(range(len(temp1)), temp1)
    plt.plot(range(len(temp2)), temp2)
    plt.plot(range(len(temp3)), temp3)
    plt.plot(range(len(temp4)), temp4)
    plt.xlabel('per 10 epoch')
    plt.ylabel('train_loss')
    plt.legend(["QSGD", "Rand-k", "LGC-SGD", "SGD"], loc='best')
    plt.savefig('./save/all_curve/fed_{}_{}_ep{}_C{}_iid{}_momentum{}_loss.pdf'.format(dataset, model,  args.epochs, args.frac, args.iid, args.momentum))
    plt.show()

if __name__ == "__main__":
    args = args_parser()
    plot_acc(args)
