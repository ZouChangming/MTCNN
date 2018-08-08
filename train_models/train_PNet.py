#coding:utf-8
from mtcnn_model import P_Net
from train import train


def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = P_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #data path
    base_dir = '../data/imglists/PNet'
    model_path = '../data/MTCNN_model/PNet/'
            
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.01
    train_PNet(base_dir, prefix, end_epoch, display, lr)
