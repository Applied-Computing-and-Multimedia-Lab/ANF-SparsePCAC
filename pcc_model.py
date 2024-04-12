import torch
import MinkowskiEngine as ME
import numpy as np

from autoencoder import Encoder, Decoder, HPEncoder, HPDecoder, PC_DeQuantizationModule, get_coordinate
from loss import get_mse, get_mse_F
from functional import bound

# this code is from "Variational image compression with a scale hyperprior" paper


class PCCModel(torch.nn.Module):
    def __init__(self, hyper=True, use_mean=True, use_QE=False, use_context=False):
        super().__init__()
        self.encoder = Encoder(channels=[3, 64, 128])
        self.decoder = Decoder(channels=[3, 64, 128])
        self.encoder2 = Encoder(channels=[3, 64, 128])
        self.decoder2 = Decoder(channels=[3, 64, 128])
        self.get_coordinate = get_coordinate()

        self.hyper = hyper
        if self.hyper:
            from entropy_model2 import __CONDITIONS__, EntropyBottleneck
            self.hpenc = HPEncoder(channels=[128, 128, 128])
            self.hpdec = HPDecoder(channels=[256, 256, 128])
            self.conditional_bottleneck = __CONDITIONS__['Gaussian'](use_mean=use_mean)
            # self.conditional_bottleneck = __CONDITIONS__['Laplacian'](use_mean=use_mean)
            # self.conditional_bottleneck = __CONDITIONS__['Logistic'](use_mean=use_mean)
            self.entropy_bottleneck = EntropyBottleneck(128)
        else:
            from entropy_model import EntropyBottleneck
            self.entropy_bottleneck = EntropyBottleneck(128)

        self.use_context = use_context
        if self.use_context:
            self.conditional_bottleneck = ContextModel(
                128, 256, self.conditional_bottleneck)

        if use_QE:
            self.QE = PC_DeQuantizationModule(channels=[3,64,3], num_layers=6)
        else:
            self.QE = None

    def get_likelihood(self, data, lossless_test):
        if self.hyper:
            h_list = self.hpenc(data)
            h = h_list


            data_hf, h_likelihood = self.entropy_bottleneck(h.F)
            data_hq = ME.SparseTensor(
                features=data_hf,
                coordinate_map_key=h.coordinate_map_key,
                coordinate_manager=h.coordinate_manager,
                device=h.device)
            if lossless_test:
                data_hq = h

            # Decoder
            out = self.hpdec(data_hq)
            if self.use_context:
                data_F, likelihood = self.conditional_bottleneck(data, condition=out.F)
            else:
                data_F, likelihood = self.conditional_bottleneck(data.F, condition=out.F)

            data_Q = ME.SparseTensor(
                features=data_F,
                coordinate_map_key=data.coordinate_map_key,
                coordinate_manager=data.coordinate_manager,
                device=data.device)
            return data_Q, (likelihood, h_likelihood)

        else:
            data_F, likelihood = self.entropy_bottleneck(data.F)
            data_Q = ME.SparseTensor(
                features=data_F,
                coordinate_map_key=data.coordinate_map_key,
                coordinate_manager=data.coordinate_manager,
                device=data.device)

            return data_Q, likelihood

    def forward(self, x, training=True, lossless_test=False):

        # Encoder
        y_list = self.encoder(x)
        y1 = y_list[0]
        ground_truth_list = y_list[1:] + [x]

        # Decoder
        out_cls_list, out = self.decoder(y1)
        x1 = x - out

        # Encoder2
        y_list2 = self.encoder2(x1)
        y2 = y_list2[0] + y1
        ground_truth_list2 = y_list2[1:] + [x1]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y2, lossless_test)
        if lossless_test:
            y_q = y2

        # Backward Decoder2
        out_cls_list2, out = self.decoder2(y_q)
        x2 = x1 - out
        if lossless_test:
            x_b = out + x2
        else:
            x_b = out

        # mse1 = get_mse_F(x_b, x1) / float(x_b.__len__())

        # Backward Encoder2
        y_list2_b = self.encoder2(x_b)
        yr = y_q - y_list2_b[0]

        # mse2 = get_mse_F(yr, y1) / float(yr.__len__())

        # Backward Decoder
        out_cls_list3, out = self.decoder(yr)

        out = x_b + out

        if self.QE is not None:
            out = self.QE(out)

        F = bound(out.F, 0, 255)
        out0 = ME.SparseTensor(
            features=F,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
            device=out.device)


        # mse3 = get_mse_F(out0, x) / float(x.__len__())

        return {'out': out0,
                'x1': x1,
                'x2': x2,
                'out_cls_list': out_cls_list,
                'out_cls_list2': out_cls_list2,
                'out_cls_list3': out_cls_list3,
                'prior': y_q,
                'likelihood': likelihood,
                'ground_truth_list': ground_truth_list,
                'ground_truth_list2': ground_truth_list2
                }


if __name__ == '__main__':
    model = PCCModel()
    print(model)

