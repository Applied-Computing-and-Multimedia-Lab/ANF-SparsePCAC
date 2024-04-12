import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader
from pcc_model import PCCModel
from trainer import Trainer
import math

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default='./Scannet dataset/scan/data_for_train/point64/') # training dataset
    parser.add_argument("--dataset_num", type=int, default=5e4) #cube 5e4

    parser.add_argument("--colar_mode", type=int, default=0, help="0 = RGB"
                                                                  "1 = YUV")
    parser.add_argument("--Loss_function_mode", type=int, default=3, help="1 = multi-scale-loss"
                                                                          "2 = RGB-loss"
                                                                          "3 = YUV-loss"
                                                                          "4 = YUV2-loss")

    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")
    parser.add_argument("--lamda", type=float, default=0.05, help="weights for total distoration.") #high rate lamba
    # parser.add_argument("--lamda", type=float, default=0.025, help="weights for total distoration.")
    parser.add_argument("--gama", type=float, default=0.01, help="weights for x2.")
    parser.add_argument("--D_scale", type=list, default=[0, 0, 1, 1], help="different distoration's parameter.")
    parser.add_argument("--mse_scale", type=list, default=[1, 1, 1], help="different mse's parameter.")
    parser.add_argument("--mse_scale1", type=list, default=[1, 1, 1], help="different mse's parameter.")
    parser.add_argument("--mse_scale2", type=list, default=[1, 1, 1], help="different mse's parameter.")
    parser.add_argument("--mse_scale3", type=list, default=[1, 1, 1], help="different mse's parameter.")

    # parser.add_argument("--init_ckpt", default='/ANFPCAC/ckpts/final_result/R7.pth') #If you want to train low rate check point open this , load the high rate check point
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=2e-4) #learning rate
    parser.add_argument("--batch_size", type=int, default=32) #batch size
    parser.add_argument("--epoch", type=int, default=500) #train epoch
    parser.add_argument("--check_time", type=float, default=40,  help='frequency for recording state (min).')
    parser.add_argument("--prefix", type=str, default='0222', help="prefix of checkpoints/logger, etc.") #the check point save loaction

    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, lamda, beta, gama, D_scale, mse_scale, mse_scale1, mse_scale2, mse_scale3, lr, check_time, Loss_function_mode, colar_mode):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.colar_mode = colar_mode
        self.Loss_function_mode = Loss_function_mode
        self.lamda = lamda
        self.beta = beta
        self.gama = gama
        self.D_scale = D_scale
        self.mse_scale = mse_scale
        self.mse_scale1 = mse_scale1
        self.mse_scale2 = mse_scale2
        self.mse_scale3 = mse_scale3
        self.lr = lr
        self.check_time = check_time


if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix),
                            ckptdir=os.path.join('./ckpts', args.prefix),
                            init_ckpt=args.init_ckpt,
                            colar_mode=args.colar_mode,
                            Loss_function_mode=args.Loss_function_mode,
                            lamda=args.lamda,
                            beta=args.beta,
                            gama=args.gama,
                            D_scale = args.D_scale,
                            mse_scale = args.mse_scale,
                            mse_scale1 = args.mse_scale1,
                            mse_scale2 = args.mse_scale2,
                            mse_scale3 = args.mse_scale3,
                            lr=args.lr,
                            check_time=args.check_time)
    # model
    model = PCCModel()

    count_model_total = 0
    for p in model.parameters():
        count_model_total += p.numel()
    print("model size == ", count_model_total)
    # trainer
    trainer = Trainer(config=training_config, model=model)

    filedirs = sorted(glob.glob(args.dataset+'*.h5'))[:int(args.dataset_num)]
    train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

    # training
    for epoch in range(0, args.epoch):
        if epoch % 50 == 0: trainer.config.lr = max(trainer.config.lr / 2, 1e-6)# update lr
        trainer.train(train_dataloader)
        sum_loss = trainer.test(test_dataloader, 'Test')
        if math.isnan(sum_loss):
            print("ERROR at sum_loss = nan")
            break
        elif epoch>0:
            if sum_loss < per_sum_loss:
                trainer.save_best_model()
        per_sum_loss = sum_loss
