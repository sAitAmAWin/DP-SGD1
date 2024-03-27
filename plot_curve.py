import matplotlib.pyplot as plt
import torch
from utils.options import args_parser
import matplotlib
from matplotlib.pyplot import MultipleLocator
matplotlib.use('TkAgg')
def plot_acc(args) :
    dataset = "mnist"
    model = "mlp"
    epsilon = 0.5
    enc1 = torch.load('./save/train_data/algorithm_ADP-SGD_dataset_{}_model_{}_epsilon{}.txt'.format(dataset,model,epsilon))
    enc2 = torch.load('./save/train_data/algorithm_DP-SGD_dataset_{}_model_{}_epsilon{}.txt'.format(dataset,model,epsilon))
    enc3 = torch.load('./save/train_data/algorithm_SGD_dataset_{}_model_{}_epsilon{}.txt'.format(dataset,model,0.5))
    enc4 = torch.load('./save/train_data/algorithm_AdaDP-SGD_dataset_{}_model_{}_epsilon{}.txt'.format(dataset, model, epsilon))
    temp1 = list(enc1)
    temp2 = list(enc2)
    temp3 = list(enc3)
    temp4 = list(enc4)
   #  plt.plot(range(len(temp1))[1:5:200], temp1[1:5:200])
    # plt.plot(range(len(temp2))[1:5:200], temp2[1:5:200])
    # plt.plot(range(len(temp3))[1:5:200], temp3[1:5:200])
    # plt.plot(range(len(temp4))[1:5:200], temp4[1:5:200])
    plt.plot(range(1, len(temp1), 10), temp1[1: 200: 10])
    plt.plot(range(1, len(temp1), 10), temp2[1: 200: 10])
    plt.plot(range(1, len(temp1), 10), temp3[1: 200: 10])
    plt.plot(range(1, len(temp1), 10), temp4[1: 200: 10])
    # x_major_locator = MultipleLocator(10)  # x轴刻度线间隔
    y_major_locator = MultipleLocator(10)  # 有y轴刻度线间隔
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel('train iteration')
    plt.ylabel('test_accuracy')
    plt.legend(["AdaDP-SGD", "DP-SGD",  "SGD", "DyDP-SGD"], loc='best')
    plt.savefig('./save/all_curve/fed_{}_{}_ep{}_C{}_epsilon{}_momentum{}.pdf'.format(dataset, model,  args.epochs, args.frac, epsilon, args.momentum))
    plt.show()

if __name__ == "__main__":
    args = args_parser()
    plot_acc(args)
