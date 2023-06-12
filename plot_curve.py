import matplotlib.pyplot as plt
import torch
from utils.options import args_parser

def plot_acc(args) :
    enc1 = torch.load('./save/train_data/algorithm_quan_dataset_cifar_model_resnet18.txt')
    enc2 = torch.load('./save/train_data/algorithm_spar_dataset_cifar_model_resnet18.txt')
    enc3 = torch.load('./save/train_data/algorithm_adaptiveQSGD_dataset_cifar_model_resnet18.txt')
    enc4 = torch.load('./save/train_data/algorithm_SGD_dataset_cifar_model_resnet18.txt')
    temp1 = list(enc1)
    temp2 = list(enc2)
    temp3 = list(enc3)
    temp4 = list(enc4)
    plt.plot(range(len(temp1)), temp1)
    plt.plot(range(len(temp2)), temp2)
    plt.plot(range(len(temp3)), temp3)
    plt.plot(range(len(temp4)), temp4)
    plt.xlabel('per 10 iter')
    plt.ylabel('test_accuracy')
    plt.legend(["quan", "spar", "adaptiveQSGD", "SGD"], loc='best')
    plt.savefig('./save/all_curve/fed_{}_{}_ep{}_C{}_iid{}_momentum{}.png'.format(args.dataset, args.model,  args.epochs, args.frac, args.iid, args.momentum))
    plt.show()

if __name__ == "__main__":
    args = args_parser()
    plot_acc(args)