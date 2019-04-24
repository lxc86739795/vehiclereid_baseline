import argparse
import os, sys
import shutil
import time
import numpy as np
import os
import os.path as osp

import math

parser = argparse.ArgumentParser(description='PyTorch Relationship')

parser.add_argument('--querylist', default='./list/veri_query_list.txt', type=str, metavar='DIR', help='path to query list')
parser.add_argument('--gallerylist', default='./list/veri_test_list.txt', type=str, metavar='DIR', help='path to gallery list')
parser.add_argument('--tracklist', default='./list/veri_test_track_list.txt', type=str, metavar='DIR', help='path to tracck list')
parser.add_argument('--queryFeat', default='./results/veri/resnet50/queryFeat.npy', type=str, metavar='DIR', help='path to query feature')
parser.add_argument('--galleryFeat', default='./results/veri/resnet50/galleryFeat.npy', type=str, metavar='DIR', help='path to gallery feature')

parser.add_argument('--dataset',  default='veri', type=str,
                    help='dataset name veri or aic (default: veri)')
parser.add_argument('--save_dir', default='./results/', type=str,
                    help='save_dir')
parser.add_argument('--TopK',default=100, type=int,
                    help='save top K indexes of results for each query (default: 100)')
                   
def main():
    global args
    args = parser.parse_args()
    print (args)
    # Create dataloader
    print ('====> Creating dataloader...')
    
    query_list = args.querylist
    gallery_list = args.gallerylist
    track_list = args.tracklist

    query_lines = open(query_list).readlines()
    gallery_lines = open(gallery_list).readlines()
    track_lines = open(track_list).readlines()
    query_feat_mat = np.load(args.queryFeat)
    gallery_feat_mat = np.load(args.galleryFeat)

    print('Query num: ', len(query_lines))
    print('Gallery num: ', len(gallery_lines))
    print('Track num: ', len(track_lines))
    print('Query feat: ', query_feat_mat.shape)
    print('Gallery feat: ', gallery_feat_mat.shape)

    track_feat_mat = None
    distmat = None

    query_names = []
    query_pids = []
    query_camids = []

    gallery_names = []

    track_pids = []
    track_camids = []

    for query_line in query_lines:
        line = query_line.strip()
        query_names.append(line)
        query_pids.append(int(line.split('_')[0]))
        query_camids.append(int(line.split('_')[1].split('c')[1]))

    for gallery_line in gallery_lines:
        line = gallery_line.strip()
        gallery_names.append(line)

    for i, track_line in enumerate(track_lines):
        #if i == 10:
        #    break
        line = track_line.strip()
        # print('Track : ', i)
        track_pids.append(int(line.split(' ')[0].split('_')[0]))
        track_camids.append(int(line.split(' ')[0].split('_')[1].split('c')[1]))
        images = line.split(' ')[1:-1]
        track_indexes = []
        for _, image in enumerate(images):
            idx = gallery_names.index(image.strip())
            track_indexes.append(idx)
        track_feats = gallery_feat_mat[track_indexes,:]
        track_feat = np.mean(track_feats, axis=0, keepdims=True)     
        if i == 0:
            track_feat_mat = track_feat
            continue
        track_feat_mat = np.concatenate((track_feat_mat, track_feat))
    print('Track feat mat: ', track_feat_mat.shape)
    np.savetxt('trackfeatmat.txt', track_feat_mat, fmt='%.4f')
    distmat = np.zeros((query_feat_mat.shape[0], track_feat_mat.shape[0]))
    distmat = (getNormMatrix(query_feat_mat, track_feat_mat.shape[0]).T + getNormMatrix(track_feat_mat, query_feat_mat.shape[0]) - 2 * np.dot(query_feat_mat, track_feat_mat.T))
    print('Dist mat: ', distmat.shape)
    np.savetxt('trackdistmat.txt', distmat, fmt='%.4f')

    cmc, mAP = eval_func(distmat, query_pids, query_camids, track_pids, track_camids, max_rank=len(track_pids))

    print('mAP : ', mAP)
    print('Rank-1 : ', cmc[0])

    return

def eval_func(distmat, q_pids, q_camids, g_pids, g_camids, max_rank=100):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print(q_pids.shape)
    print(q_camids.shape)
    print(g_pids.shape)
    print(g_camids.shape)

    num_q, num_g = distmat.shape
    max_rank = args.TopK
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print('Saving resulting indexes...', indices.shape)
    np.save('result.npy', indices[:, :args.TopK])
    np.savetxt('result.txt', indices[:, :args.TopK], fmt='%d')
    if args.dataset == 'aic':
        return None, None

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def getNormMatrix(x, lines_num):
    """
    Get a lines_num x size(x, 1) matrix
    """ 
    return np.ones((lines_num, 1)) * np.sum(np.square(x), axis = 1)

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__=='__main__':
    main()
