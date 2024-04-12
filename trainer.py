import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
from entropy_model2 import estimate_bpp, estimate_bpp_eval
from loss import get_mse, get_mse_F, get_bits, psnr, yuv_psnr, rgb2yuv, yuv2rgb
from mse_loss import all_scale_loss, RGB_loss, YUV_loss ,YUV2_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter
import csv


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'D0': [], 'D1': [], 'D2': [], 'D_hat': [], 'mses': [], 'y2': [], 'bpp': [], 'sum_loss': [], 'yuv-psnr': []}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.init_ckpt == '':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()},
                   os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return

    def save_best_model(self):
        torch.save({'model': self.model.state_dict()},
                   os.path.join(self.config.ckptdir, "best.pth"))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params": self.model._modules[module_name].parameters(), 'lr': self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step, mode=''):
        # print record
        self.logger.info('=' * 10 + main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items():
            self.record_set[k] = np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items():
            self.logger.info(k + ': ' + str(np.round(v, 4).tolist()))

        if mode == 'train':
            ckpt = self.config.ckptdir + '/train.csv'
        elif mode == 'test':
            ckpt = self.config.ckptdir + '/test.csv'
        if self.epoch == 0:
            with open(ckpt, 'w+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["epoch", "D'", "D1", "D2", "D_hat", "y2", "bpp", "sum_Loss", "y_psnr", "u_psnr", "v_psnr"])
                writer.writerow(
                    [self.epoch, self.record_set['D0'], self.record_set['D1'], self.record_set['D2'], self.record_set['D_hat'], self.record_set['y2'], self.record_set['bpp'], self.record_set['sum_loss'],
                     self.record_set['yuv-psnr'][0][0], self.record_set['yuv-psnr'][0][1],
                     self.record_set['yuv-psnr'][0][2]])
        else:
            with open(ckpt, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [self.epoch, self.record_set['D0'], self.record_set['D1'], self.record_set['D2'], self.record_set['D_hat'], self.record_set['y2'], self.record_set['bpp'], self.record_set['sum_loss'],
                     self.record_set['yuv-psnr'][0][0], self.record_set['yuv-psnr'][0][1],
                     self.record_set['yuv-psnr'][0][2]])
        # return zero
        for k in self.record_set.keys():
            self.record_set[k] = []

        return

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        total_loss = 0
        for _, (coords, feats, attri) in enumerate(tqdm(dataloader)):
            # data

            x = ME.SparseTensor(features=attri.float(), coordinates=coords, device=device)

            if self.config.colar_mode == 0:
                pass
            if self.config.colar_mode == 1:
                x_F = x.F
                x_F, _ = rgb2yuv(x_F, x_F)
                x = ME.SparseTensor(features=x_F,
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager,
                                    device=x.device)

            # # Forward.
            out_set = self.model(x, training=False)

            # # loss
            if self.config.Loss_function_mode == 1:
                mse_list, all_mse_list = all_scale_loss(self.config.mse_scale, self.config.mse_scale1, self.config.mse_scale2, x, out_set)
            if self.config.Loss_function_mode == 2:
                mse_list, all_mse_list = RGB_loss(x, out_set)
            if self.config.Loss_function_mode == 3:
                mse_list, all_mse_list = YUV_loss(self.config.mse_scale, self.config.mse_scale1, self.config.mse_scale2, self.config.mse_scale3, x, out_set)
            if self.config.Loss_function_mode == 4:
                mse_list, all_mse_list = YUV2_loss(self.config.mse_scale1, x, out_set)

            x2_loss = abs(out_set['x2'].F).pow(2).mean()
            x1_loss = abs(out_set['x1'].F).pow(2).mean()

            nums_list = x.F.shape[0]
            bpp = estimate_bpp(out_set['likelihood'][0], nums_list, input=x.F).sum()
            bpp += estimate_bpp(out_set['likelihood'][1], nums_list, input=x.F).sum()

            # diff_D_scale
            D_scale = self.config.D_scale
            sum_loss = self.config.beta * bpp + \
                       self.config.lamda * \
                       ((D_scale[0] * mse_list[0] + D_scale[1] * mse_list[1] + D_scale[2] * mse_list[2] + D_scale[3] * mse_list[3]) +
                        self.config.gama * x2_loss)

            # back to rgb
            if self.config.colar_mode == 0:
                pass
            if self.config.colar_mode == 1:
                x_F = x.F
                x_F, _ = yuv2rgb(x_F, x_F)
                x = ME.SparseTensor(features=x_F,
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager,
                                    device=x.device)
                out_F = out_set['out'].F
                out_F, _ = yuv2rgb(out_F, out_F)
                out_set['out'] = ME.SparseTensor(features=out_F,
                                                 coordinate_map_key=out_set['out'].coordinate_map_key,
                                                 coordinate_manager=out_set['out'].coordinate_manager,
                                                 device=out_set['out'].device)

            metrics = []
            metrics.append(yuv_psnr(out_set['out'].F.cpu().numpy(), x.F.cpu().numpy()))

            # record
            self.record_set['D0'].append(mse_list[0].item())
            self.record_set['D1'].append(mse_list[1].item())
            self.record_set['D2'].append(mse_list[2].item())
            self.record_set['D_hat'].append(mse_list[3].item())
            self.record_set['mses'].append(all_mse_list)
            self.record_set['y2'].append(x2_loss.item())
            self.record_set['bpp'].append(bpp.item())
            self.record_set['sum_loss'].append(sum_loss.item())
            self.record_set['yuv-psnr'].append(metrics)
            total_loss = total_loss + sum_loss
            torch.cuda.empty_cache()  # empty cache.
        self.record(main_tag=main_tag, global_step=self.epoch, mode='test')
        self.epoch += 1
        return total_loss

    def train(self, dataloader):
        self.logger.info('=' * 40 + '\n' + 'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('Colar_mode:' + str(self.config.colar_mode))
        self.logger.info('Loss_function_mode:' + str(self.config.Loss_function_mode))
        self.logger.info('lamda:' + str(self.config.lamda) + '\tgama:' + str(self.config.gama))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        self.logger.info('D_scale:' + str(self.config.D_scale))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))

        # start_time = time.time()
        for batch_step, (coords, feats, attri) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            x = ME.SparseTensor(features=attri.float(), coordinates=coords, device=device)

            if self.config.colar_mode == 0:
                pass
            if self.config.colar_mode == 1:
                x_F = x.F
                x_F, _ = rgb2yuv(x_F, x_F)
                x = ME.SparseTensor(features=x_F,
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager,
                                    device=x.device)

            # forward
            out_set = self.model(x, training=True)

            # loss
            if self.config.Loss_function_mode == 1:
                mse_list, all_mse_list = all_scale_loss(self.config.mse_scale, self.config.mse_scale1, self.config.mse_scale2, x, out_set)
            if self.config.Loss_function_mode == 2:
                mse_list, all_mse_list = RGB_loss(x, out_set)
            if self.config.Loss_function_mode == 3:
                mse_list, all_mse_list = YUV_loss(self.config.mse_scale, self.config.mse_scale1, self.config.mse_scale2, self.config.mse_scale3, x, out_set)
            if self.config.Loss_function_mode == 4:
                mse_list, all_mse_list = YUV2_loss(self.config.mse_scale1, x, out_set)

            x2_loss = abs(out_set['x2'].F).pow(2).mean()
            x1_loss = abs(out_set['x1'].F).pow(2).mean()

            nums_list = x.F.shape[0]
            bpp = estimate_bpp(out_set['likelihood'][0], nums_list, input=x.F).sum()
            bpp += estimate_bpp(out_set['likelihood'][1], nums_list, input=x.F).sum()

            # diff_D_scale
            D_scale = self.config.D_scale
            sum_loss = self.config.beta * bpp + \
                       self.config.lamda * \
                       ((D_scale[0] * mse_list[0] + D_scale[1] * mse_list[1] + D_scale[2] * mse_list[2] + D_scale[3] * mse_list[3]) +
                        self.config.gama * x2_loss)

            # backward & optimize
            sum_loss.backward()
            self.optimizer.step()

            # back to rgb
            if self.config.colar_mode == 0:
                pass
            if self.config.colar_mode == 1:
                x_F = x.F
                x_F, _ = yuv2rgb(x_F, x_F)
                x = ME.SparseTensor(features=x_F,
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager,
                                    device=x.device)
                out_F = out_set['out'].F
                out_F, _ = yuv2rgb(out_F, out_F)
                out_set['out'] = ME.SparseTensor(features=out_F,
                                                coordinate_map_key=out_set['out'].coordinate_map_key,
                                                coordinate_manager=out_set['out'].coordinate_manager,
                                                device=out_set['out'].device)

            # metric & record
            with torch.no_grad():
                metrics = []
                metrics.append(yuv_psnr(out_set['out'].F.cpu().numpy(), x.F.cpu().numpy()))

                self.record_set['D0'].append(mse_list[0].item())
                self.record_set['D1'].append(mse_list[1].item())
                self.record_set['D2'].append(mse_list[2].item())
                self.record_set['D_hat'].append(mse_list[3].item())
                self.record_set['mses'].append(all_mse_list)
                self.record_set['y2'].append(x2_loss.item())
                self.record_set['bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(sum_loss.item())
                self.record_set['yuv-psnr'].append(metrics)

                # if (time.time() - start_time) > self.config.check_time * 60:
                #     self.record(main_tag='Train', global_step=self.epoch * len(dataloader) + batch_step)
                #     self.save_model()
                #     start_time = time.time()
            torch.cuda.empty_cache()  # empty cache.
        with torch.no_grad():
            self.record(main_tag='Train', global_step=self.epoch * len(dataloader) + batch_step, mode='train')
        if self.epoch % 100 == 99:
            self.save_model()
        if self.epoch == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(self.config.ckptdir, "best.pth"))
        # self.epoch += 1
        return