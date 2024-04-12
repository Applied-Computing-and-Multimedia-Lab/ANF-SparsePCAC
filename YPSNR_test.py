import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud
import quality_eval
from importlib import import_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def metrics_to_dict(metrics, prefix):
    mse, psnr, mae = metrics
    return {f'{prefix}y_mse': mse[0], f'{prefix}u_mse': mse[1], f'{prefix}v_mse': mse[2],
            f'{prefix}y_mae': mae[0], f'{prefix}u_mae': mae[1], f'{prefix}v_mae': mae[2],
            f'{prefix}y_psnr': psnr[0], f'{prefix}u_psnr': psnr[1], f'{prefix}v_psnr': psnr[2]}


def run(file1, file2, point_size=1):
    # file1 = open(file1)
    # file2 = open(file2)
    assert os.path.exists(file1), f'{file1} not found'
    assert os.path.exists(file2), f'{file2} not found'

    pc1 = PyntCloud.from_file(file1)
    pc2 = PyntCloud.from_file(file2)

    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    final_metrics, fwd_metrics, bwd_metrics = quality_eval.color_with_geo(pc1.points[cols].values, pc2.points[cols].values)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='23_eval_merged.py', description='Eval a merged point cloud.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file1', help='Original file.', default='longdress.ply')
    parser.add_argument('--file2', help='Distorted file.', default='900.ply')
    ori_list = [
                './testing_data/testdata/8iVFB/longdress.ply',
                './testing_data/testdata/8iVFB/loot.ply',
                './testing_data/testdata/8iVFB/redandblack.ply',
                './testing_data/testdata/8iVFB/soldier.ply',
                './testing_data/testdata/MVUB/david.ply',
                './testing_data/testdata/MVUB/phil.ply',
                './testing_data/testdata/MVUB/sarah.ply',
                './testing_data/testdata/MVUB/andrew.ply',
                  ]

    filedir_list = [
                './recolour/recoloured/longdress_vox10_1300_r',
                './recolour/recoloured/loot_vox10_1200_r',
                './recolour/recoloured/redandblack_vox10_1550_r',
                './recolour/recoloured/soldier_vox10_0690_r',
                './recolour/recoloured/david_vox9_frame0000_r',
                './recolour/recoloured/phil_vox9_frame0139_r',
                './recolour/recoloured/sarah_vox9_frame0023_r',
                './recolour/recoloured/andrew_vox9_frame0000_r'
                  ]
    parser.add_argument("--resultdir", default='./recolor')

    parser.add_argument('--point_size', default=1, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    
    for ori_filedir, rec_filedir in  zip(ori_list, filedir_list):
        for idx in range(6):
            data = run(ori_filedir, rec_filedir + str(idx+1) +'_dec_re_attri.ply', args.point_size)
            [data.pop(key) for key in
             ['AB_u_mae', 'AB_u_mse', 'AB_u_psnr', 'AB_v_mae', 'AB_v_mse', 'AB_v_psnr', 'AB_y_mae', 'AB_y_mse',
              'AB_y_psnr',
              'BA_u_mae', 'BA_u_mse', 'BA_u_psnr', 'BA_v_mae', 'BA_v_mse', 'BA_v_psnr', 'BA_y_mae', 'BA_y_mse',
              'BA_y_psnr',
              'u_mae', 'v_mae', 'y_mae', 'qp', 'color_bits_per_input_point', 'color_bitstream_size_in_bytes']]
            results = pd.DataFrame.from_dict(data, orient='index').T
            if idx == 0:
                all_results = results.copy(deep=True)
            else:
                all_results = all_results.append(results, ignore_index=True)
            csv_name = os.path.join(args.resultdir, os.path.split(rec_filedir + str(idx+1) +'_dec_re_attri.ply')[-1].split('.')[0] + '.csv')
            all_results.to_csv(csv_name, index=False)
            print('Wrile results to: \t', csv_name)

