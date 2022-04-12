from os.path import join, exists, dirname
from sklearn.neighbors import KDTree
from tool import DataProcessing as DP
from helper_ply import write_ply
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os, pickle, argparse

def process(pc_path,dataset_path, sample_type, grid_size, test_files, train_files):
    pc_path_posix = pc_path.replace('\\','/')
    cloud_name = pc_path_posix.split('/')[-1][:-4]
    print('start to process:', cloud_name)

    # create output directory
    out_folder = join(dirname(dataset_path), sample_type + '_{:.3f}'.format(grid_size))
    os.makedirs(out_folder) if not exists(out_folder) else None

    # check if it has already calculated
    if exists(join(out_folder, cloud_name + '_KDTree.pkl')):
        print(cloud_name, 'already exists, skipped')
        return

    if pc_path in train_files:
        xyz, rgb, labels = DP.read_ply_data(pc_path, with_rgb=True)
    else:
        xyz, rgb = DP.read_ply_data(pc_path, with_rgb=True, with_label=False)
        labels = np.zeros(len(xyz), dtype=np.uint8)

    sub_ply_file = join(out_folder, cloud_name + '.ply')
    if sample_type == 'grid':
        sub_xyz, sub_rgb, sub_labels = DP.grid_sub_sampling(xyz, rgb, labels, grid_size)
    else:
        sub_xyz, sub_rgb, sub_labels = DP.random_sub_sampling(xyz, rgb, labels, random_sample_ratio)

    sub_rgb = sub_rgb / 255.0
    sub_labels = np.squeeze(sub_labels)
    write_ply(sub_ply_file, [sub_xyz, sub_rgb, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = join(out_folder, cloud_name + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
         pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(out_folder, cloud_name + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='the number of GPUs to use [default: 0]')
    FLAGS = parser.parse_args()
    dataset_name = 'SensatUrban'
    dataset_path = FLAGS.dataset_path
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = 0.2
    random_sample_ratio = 10
    train_files = np.sort([join(dataset_path, 'train', i) for i in os.listdir(join(dataset_path, 'train'))])
    test_files = np.sort([join(dataset_path, 'test', i) for i in os.listdir(join(dataset_path, 'test'))])
    files = np.hstack((train_files,test_files))
    thread_num = 16
    for sample_type in preparation_types:
        mp.freeze_support()
        po = Pool(thread_num)
        for pc_path in files:
            po.apply_async(process, (pc_path, dataset_path, sample_type, grid_size, test_files,train_files))
            
        po.close()
        po.join()
        po = Pool(thread_num)
