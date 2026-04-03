import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser('Stitch MIST dataset pairs side by side')
parser.add_argument('--stain', type=str, required=True, help='Stain type: HER2 | ER | Ki67 | PR')
parser.add_argument('--mist_root', type=str, default='./datasets/MIST_raw', help='Path to MIST root directory')
parser.add_argument('--output_root', type=str, default='./datasets', help='Output directory root')
parser.add_argument('--num_imgs', type=int, default=1000000, help='Max images per split')
args = parser.parse_args()

splits = [
    ('trainA', 'trainB', 'train'),
    ('valA',   'valB',   'test'),
]

for src_A_name, src_B_name, dst_name in splits:
    src_A = os.path.join(args.mist_root, args.stain, 'TrainValAB', src_A_name)
    src_B = os.path.join(args.mist_root, args.stain, 'TrainValAB', src_B_name)
    dst   = os.path.join(args.output_root, f'MIST_{args.stain}', dst_name)

    os.makedirs(dst, exist_ok=True)

    images = [f for f in sorted(os.listdir(src_A)) if not f.startswith('.')]
    images = images[:args.num_imgs]

    print(f'split = {dst_name}, stitching {len(images)} images...')
    for img_name in images:
        img_A = cv2.imread(os.path.join(src_A, img_name))
        img_B = cv2.imread(os.path.join(src_B, img_name))
        stitched = np.concatenate([img_A, img_B], axis=1)
        cv2.imwrite(os.path.join(dst, img_name), stitched)

    print(f'split = {dst_name}, done -> {dst}')