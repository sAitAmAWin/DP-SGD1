import matplotlib.pyplot as plt
import torch
from utils.options import args_parser
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use('TkAgg')
def plot_acc(args) :
    dataset = "mnist"
    model = "mlp"
    epsilon = 0.1
    enc1 = torch.load('./save/train_data/algorithm_ADP-SGD_lr{}_ep{}_dataset_{}_model_{}_epsilon{}_acc.txt'.format(args.lr, args.epochs, dataset, model, epsilon))
    enc2 = torch.load('./save/train_data/algorithm_DP-SGD_lr{}_ep{}_dataset_{}_model_{}_epsilon{}_acc.txt'.format(args.lr, args.epochs, dataset, model, epsilon))
    enc3 = torch.load('./save/train_data/algorithm_SGD_dataset_{}_model_{}_epsilon{}.txt'.format(dataset, model, 0.5))
    enc4 = torch.load('./save/train_data/algorithm_AdaDP-SGD_lr{}_ep{}_dataset_{}_model_{}_epsilon{}_acc.txt'.format(args.lr, args.epochs, dataset, model, epsilon))
    temp1 = list(enc1)
    temp2 = list(enc2)
    temp3 = list(enc3)
    temp4 = list(enc4)
    print(temp1[-1])
    print(temp2[-1])
    print(temp3[-1])
    print(temp4[-1])
    #plt.plot(range(9, len(temp1), 10), temp1[9: len(temp1): 10])
    #plt.plot(range(9, len(temp1), 10), temp2[9: len(temp1): 10])
    #plt.plot(range(9, len(temp1), 10), temp3[9: len(temp1): 10])
    #plt.plot(range(9, len(temp1), 10), temp4[9: len(temp1): 10])
    plt.plot(range(len(temp1)), temp1)
    plt.plot(range(len(temp2)), temp2)
    plt.plot(range(len(temp1)), temp3[0: len(temp1)])
    plt.plot(range(len(temp4)), temp4)
    #y_major_locator = MultipleLocator(10)  # 有y轴刻度线间隔
    #ax = plt.gca()
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('train iteration')
    plt.ylabel('test_accuracy')
    plt.legend(["AdaptiveDP-SGD", "DP-SGD",  "SGD", "DynamicDP-SGD"], loc='best')
    plt.savefig('./save/all_curve/fed_{}_{}_ep{}_C{}_epsilon{}_momentum{}.pdf'.format(dataset, model,  args.epochs, args.frac, epsilon, args.momentum))
    plt.show()

if __name__ == "__main__":
    args = args_parser()
    plot_acc(args)
