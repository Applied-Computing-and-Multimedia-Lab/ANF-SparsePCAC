import torch
import numpy as np
import os
from pcc_model import PCCModel
from coder import Coder
import time
from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo, write_ply_ascii_attri
from pc_error import pc_error, pc_error_rgb
from YPSNR_test import run
import pandas as pd
from loss import rgb2yuv, yuv2rgb
import MinkowskiEngine as ME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(filedir_list, ckptdir_list, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024, colar_mode=0):
    for filedir in  filedir_list:
        # load data
        start_time = time.time()
        x = load_sparse_tensor(filedir, device)
        x = x.float()
        if colar_mode == 0:
            pass
        if colar_mode == 1:
            x_F = x.F
            x_F, _ = rgb2yuv(x_F, x_F)
            x = ME.SparseTensor(features=x_F,
                                coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager,
                                device=x.device)
        print('=' * 10, os.path.split(filedir)[-1].split('.')[0], '=' * 10)
        print('Loading Time:\t', round(time.time() - start_time, 4), 's')

        # output filename
        if not os.path.exists(outdir): os.makedirs(outdir)
        filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
        print('output filename:\t', filename)

        # load model
        model = PCCModel().to(device)

        for idx, ckptdir in enumerate(ckptdir_list):
            print('=' * 10, idx + 1, '=' * 10)
            # load checkpoints
            assert os.path.exists(ckptdir)
            ckpt = torch.load(ckptdir)
            model.load_state_dict(ckpt['model'])
            print('load checkpoint from \t', ckptdir)
            coder = Coder(model=model, filename=filename)

            # postfix: rate index
            postfix_idx = '_r' + str(idx + 1)

            # down-scale
            if scaling_factor != 1:
                x_in = scale_sparse_tensor(x, factor=scaling_factor)
            else:
                x_in = x

            # get_coordinate
            _, main_coord = model.get_coordinate(x_in)
            hyper_coord, _ = model.get_coordinate(main_coord)

            # encode
            start_time = time.time()
            _, _, _, _, _ = coder.encode(x_in, postfix=postfix_idx)
            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)

            # decode
            start_time = time.time()
            x_dec = coder.decode(postfix=postfix_idx, rho=rho, y_key=main_coord.coordinate_map_key,
                                 y_manager=main_coord.coordinate_manager, h_key=hyper_coord.coordinate_map_key,
                                 h_manager=hyper_coord.coordinate_manager)
            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)

            # up-scale
            if scaling_factor != 1:
                x_dec = scale_sparse_tensor(x_dec, factor=1.0 / scaling_factor)

            # bitrate
            bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                             for postfix in ['_F.bin', '_H.bin', '_h_hat_F.bin', '_h_hat_H.bin']])

            bpps = (bits / len(x)).round(3)
            start_time = time.time()

            # back to rgb
            if colar_mode == 0:
                pass
            if colar_mode == 1:
                x_dec_F = x_dec.F
                x_dec_F, _ = yuv2rgb(x_dec_F, x_dec_F)
                x_dec = ME.SparseTensor(features=x_dec_F,
                                        coordinate_map_key=x_dec.coordinate_map_key,
                                        coordinate_manager=x_dec.coordinate_manager,
                                        device=x_dec.device)

            write_ply_ascii_attri(filename + postfix_idx + '_dec_attri.ply', x_dec.C.detach().cpu().numpy()[:, 1:],
                                  x_dec.F.detach().cpu().numpy())
            print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

            start_time = time.time()
            data = run(filedir, filename + postfix_idx + '_dec_attri.ply', point_size='')
            
            # save results
            YUV_PSNR = data[1]
            result = {'bits': sum(bits).round(3), 'bpp': sum(bpps).round(3), 'Y-PSNR': YUV_PSNR[0], 'U-PSNR': YUV_PSNR[1], 'V-PSNR': YUV_PSNR[2], 'time(enc)': time_enc, 'time(dec)': time_dec}
            results = pd.DataFrame.from_dict(result, orient='index').T

            if idx == 0:
                all_results = results.copy(deep=True)
            else:
                all_results = all_results.append(results, ignore_index=True)
            csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0] + '.csv')
            all_results.to_csv(csv_name, index=False)
            print('Wrile results to: \t', csv_name)

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ============== test file  =================
    #input the orignal point cloud path
    filedir_list = [
        './testdata/8iVFB/longdress_vox10_1300.ply',
        './testdata/8iVFB/loot_vox10_1200.ply',
        './testdata/8iVFB/redandblack_vox10_1550.ply',
        './testdata/8iVFB/soldier_vox10_0690.ply',
                  ]

    # ====== Output file && check point =========
    Output = '/0222'
    Ckpt = '/final_result'
    # ============================================

    parser.add_argument("--colar_mode", default=0, help="0 = RGB"
                                                        "1 = YUV")
    # parser.add_argument("--outdir", default='./output')
    # parser.add_argument("--resultdir", default='output')
    parser.add_argument("--outdir", default='./output' + Output)
    parser.add_argument("--resultdir", default='./output' + Output)
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0,
                        help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)

    # ckptdir_list = ['./ckpts' + Ckpt + '/best.pth']

    ckptdir_list = [
        './ckpts' + Ckpt + '/R0.pth',
        './ckpts' + Ckpt + '/R1.pth',
        './ckpts' + Ckpt + '/R2.pth',
        './ckpts' + Ckpt + '/R3.pth',
        './ckpts' + Ckpt + '/R4.pth',
        './ckpts' + Ckpt + '/R5.pth',
        './ckpts' + Ckpt + '/R6.pth',
        './ckpts' + Ckpt + '/R7.pth',
    ]

    all_results = test(filedir_list, ckptdir_list, args.outdir, args.resultdir, scaling_factor=args.scaling_factor,
                       rho=args.rho, res=args.res, colar_mode=args.colar_mode)