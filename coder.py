import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
from functional import bound

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo, write_ply_ascii_attri

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

from pcc_model import PCCModel


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """

    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)
        gpcc_encode(self.ply_filename, self.filename + postfix + '_C.bin')
        # os.system('rm '+self.ply_filename)

        return

    def decode(self, postfix=''):
        gpcc_decode(self.filename + postfix + '_C.bin', self.ply_filename)
        coords, feats = read_ply_ascii_geo(self.ply_filename)
        # os.system('rm '+self.ply_filename)

        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """

    def __init__(self, filename, model, entropy_model):
        self.filename = filename
        self.model = model
        self.entropy_model = entropy_model.cpu()

    def encode_ori(self, feats, postfix=''):
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename + postfix + '_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename + postfix + '_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
        return feats

    def encode(self, strings, shape, postfix=''):
        with open(self.filename + postfix + '_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename + postfix + '_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
        return

    def decode_ori(self, postfix=''):
        with open(self.filename + postfix + '_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename + postfix + '_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        return feats

    def decode(self, postfix='', condition=None, y_key='', y_manager=''):
        with open(self.filename + postfix + '_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename + postfix + '_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
        shapelist = (shape[0], shape[1])
        if condition == None:
            feats = self.entropy_model.decompress(strings, shapelist)
        else:
            if self.model.use_context:
            # ==== Context ===
                feats = self.entropy_model.decompress(strings, shapelist, condition=condition.F.cpu(), y_key=y_key, y_manager=y_manager) # 11 01
            #     feats = self.entropy_model.decompress(strings, shapelist, condition=condition.F, y_key=y_key, y_manager=y_manager) # 00 10
            else:
                feats = self.entropy_model.decompress(strings, shapelist, condition=condition.F)
        return feats


class Coder():
    def __init__(self, model, filename):
        self.model = model
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder_f = FeatureCoder(self.filename, self.model, model.entropy_bottleneck)
        self.feature_coder_c = FeatureCoder(self.filename, self.model, model.conditional_bottleneck)

    @torch.no_grad()
    def encode(self, x, postfix=''):
        # Encoder
        y_list = self.model.encoder(x)

        y1 = y_list[0]
        y_key = y_list[0].coordinate_map_key
        y_manager = y_list[0].coordinate_manager

        # Decoder
        out_cls_list, out = self.model.decoder(y1)
        x1 = x - out

        # Encoder2
        y_list2 = self.model.encoder2(x1)
        y2 = y_list2[0] + y1
        # ground_truth_list2 = y_list2[1:] + [x1]

        # Hyper Encoder
        h = self.model.hpenc(y2)
        # h = h[0]
        h_key = h.coordinate_map_key
        h_manager = h.coordinate_manager
        side_stream, h_hat = self.model.entropy_bottleneck.compress(
            h.F.cpu(), return_sym=True)
        self.feature_coder_f.encode(side_stream, h.F.shape, postfix=postfix + "_h_hat")
        # h_hat = self.feature_coder.encode_ori(h.F, postfix=postfix + "_h_hat")
        data_hq = ME.SparseTensor(
            features=h_hat,
            coordinate_map_key=h_key,
            coordinate_manager=h_manager,
            device=h.device)

        # Hyper Decoder
        condition = self.model.hpdec(data_hq)
        if self.model.use_context:
        # ===== Context ====
            stream = self.model.conditional_bottleneck.compress(
                y2, condition=condition.F.cpu(), return_sym=False) # 11 10
        #     stream = self.model.conditional_bottleneck.compress(
        #         y2.F, condition=condition.F, return_sym=False) # 00 01
        else:
            stream = self.model.conditional_bottleneck.compress(
                y2.F, condition=condition.F, return_sym=False)
        self.feature_coder_c.encode(stream, y2.F.shape, postfix=postfix)

        return y2, y_key, y_manager, h_key, h_manager

    @torch.no_grad()
    def decode(self, rho=1, postfix='', y_key='', y_manager='', h_key='', h_manager=''):
        # decode feat
        h_F = self.feature_coder_f.decode(postfix=postfix + "_h_hat")
        # h_F = self.feature_coder.decode_ori(postfix=postfix + "_h_hat")
        h = ME.SparseTensor(
            features=h_F,
            coordinate_map_key=h_key,
            coordinate_manager=h_manager,
            device=device)
        condition = self.model.hpdec(h)

        # decode feat
        y_F = self.feature_coder_c.decode(postfix=postfix, condition=condition, y_key=y_key, y_manager=y_manager)


        y = ME.SparseTensor(
            features=y_F,
            coordinate_map_key=y_key,
            coordinate_manager=y_manager,
            device=device)

        # decode label
        out_list, x1_hat = self.model.decoder2(y)
        y_list_r = self.model.encoder2(x1_hat)
        yr = y - y_list_r[0]
        __, x_hat = self.model.decoder(yr)
        # print("mu_1_rev", out.F.shape)
        out = x1_hat + x_hat

        if self.model.QE is not None:
            out = self.model.QE(out)

        F = bound(out.F, 0, 255)
        out0 = ME.SparseTensor(
            features=F,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
            device=out.device)

        return out0

    @torch.no_grad()
    def ggg(self, x, out):
        # Encoder
        result = self.model.get_ground_truth(x, out)

        return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='ckpts/r3_0.10bpp.pth')
    # parser.add_argument("--ckptdir", default='./ckpts/epoch_19.pth')
    parser.add_argument("--filedir",
                        default='/media/youqi/data/Qiujijin_pointcould/PCACv2/PCGCv2-master/testing_data/andrew.ply')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0,
                        help='the ratio of the number of output points to the number of input points')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    args = parser.parse_args()
    filedir = args.filedir

    # load data
    start_time = time.time()
    x = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)

    # model
    print('=' * 10, 'Test', '=' * 10)
    model = PCCModel().to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])
    print('load checkpoint from \t', args.ckptdir)

    # coder
    coder = Coder(model=model, filename=filename)

    # down-scale
    if args.scaling_factor != 1:
        x_in = scale_sparse_tensor(x, factor=args.scaling_factor)
    else:
        x_in = x

    # encode
    start_time = time.time()
    _ = coder.encode(x_in)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    x_dec = coder.decode(rho=args.rho)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    # up-scale
    if args.scaling_factor != 1:
        x_dec = scale_sparse_tensor(x_dec, factor=1.0 / args.scaling_factor)

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix) * 8 \
                     for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
    bpps = (bits / len(x)).round(3)
    print('bits:\t', bits, '\nbpps:\t', bpps)
    print('bits:\t', sum(bits), '\nbpps:\t', sum(bpps).round(3))

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename + '_dec.ply', x_dec.C.detach().cpu().numpy()[:, 1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(args.filedir, filename + '_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
