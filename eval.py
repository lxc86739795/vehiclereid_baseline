import argparse
import os, sys
import shutil
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
from torch.autograd import Variable
import math
from networks.resnet import resnet50, resnet101
from dataset.dataset import VeriDataset


parser = argparse.ArgumentParser(description='PyTorch Relationship')

parser.add_argument('querypath', metavar='DIR', help='path to query set')
parser.add_argument('querylist', metavar='DIR', help='path to query list')
parser.add_argument('gallerypath', metavar='DIR', help='path to gallery set')
parser.add_argument('gallerylist', metavar='DIR', help='path to gallery list')
parser.add_argument('--dataset',  default='veri', type=str,
                    help='dataset name (default: veri)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('--batch_size', '--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('-n', '--num_classes', default=576, type=int, metavar='N',
                    help='number of classes / categories')
parser.add_argument('--backbone',  default='resnet50', type=str,
                    help='backbone network resnet50 or resnet101 (default: resnet50)')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to weights (default: none)')
parser.add_argument('--scale-size',default=224, type=int,
                    help='input size')
parser.add_argument('--crop_size',default=224, type=int,
                    help='crop size')
parser.add_argument('--save_dir',default='./results/', type=str,
                    help='save_dir')
parser.add_argument('--TopK',default=100, type=int,
                    help='save top K indexes of results for each query (default: 100)')
                   

def get_dataset(dataset_name, query_dir, query_list, gallery_dir, gallery_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    if dataset_name == 'veri':
        data_transform = transforms.Compose([
                transforms.Scale((scale_size,scale_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size,crop_size)),
                transforms.ToTensor(),
                normalize])

        query_set = VeriDataset(query_dir, query_list, data_transform, is_train=False )
        gallery_set = VeriDataset(gallery_dir, gallery_list, data_transform, is_train=False )

    query_loader = DataLoader(dataset=query_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    gallery_loader = DataLoader(dataset=gallery_set, num_workers=args.workers,
                            batch_size=args.batch_size, shuffle=False)
    
    return query_loader, gallery_loader
    
def main():
    global args
    args = parser.parse_args()
    print (args)
    # Create dataloader
    print ('====> Creating dataloader...')
    
    query_dir = args.querypath
    query_list = args.querylist
    gallery_dir = args.gallerypath
    gallery_list = args.gallerylist
    dataset_name = args.dataset

    query_loader, gallery_loader = get_dataset(dataset_name, query_dir, query_list, gallery_dir, gallery_list)
    # load network
    if args.backbone == 'resnet50':
        model = resnet50(num_classes=args.num_classes)
    elif args.backbone == 'resnet101':
        model = resnet101(num_classes=args.num_classes)
    
    print(args.weights)

    if args.weights != '':
        try:
            model = torch.nn.DataParallel(model)
            ckpt = torch.load(args.weights)
            model.load_state_dict(ckpt['state_dict'])
            print ('!!!load weights success !!! path is ', args.weights)
        except Exception as e:
            print ('!!!load weights failed !!! path is ', args.weights)
            return
    else:
        print('!!!Load Weights PATH ERROR!!!')
        return
    model.cuda()
    mkdir_if_missing(args.save_dir)
    
    cudnn.benchmark = True
    evaluate(query_loader, gallery_loader, model)

    return

def evaluate(query_loader, gallery_loader, model):
    print('Start evaluation...')
    query_feats = []
    query_pids = []
    query_camids = []

    gallery_feats = []
    gallery_pids = []
    gallery_camids = []

    end = time.time()
    # switch to eval mode
    model.eval()

    print('Processing query set...')
    queryN = 0
    for i, (image, pid, camid) in enumerate(query_loader):
        # if i == 10:
        #     break
        print('Extracting feature of image '+'%d:'%i)
        query_pids.append(pid)
        query_camids.append(camid)
        image = torch.autograd.Variable(image).cuda()
        output, feat = model(image)
        query_feats.append(feat.data.cpu())
        queryN = queryN+1

    query_time = time.time()-end
    end = time.time()
    print('Processing query set... \tTime[{0:.3f}]'.format(query_time))

    print('Processing gallery set...')
    galleryN = 0
    for i, (image, pid, camid) in enumerate(gallery_loader):
        # if i == 20:
        #     break
        print('Extracting feature of image '+'%d:'%i)
        gallery_pids.append(pid)
        gallery_camids.append(camid)
        image = torch.autograd.Variable(image).cuda()
        output, feat = model(image)
        gallery_feats.append(feat.data.cpu())
        galleryN = galleryN+1

    gallery_time = time.time()-end
    print('Processing gallery set... \tTime[{0:.3f}]'.format(gallery_time))
    print('Computing CMC and mAP...')
    cmc, mAP, distmat = compute(query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids)
    print('Saving distmat...')
    np.save(args.save_dir+'distmat.npy', np.asarray(distmat))
    np.savetxt(args.save_dir+'distmat.txt', np.asarray(distmat), fmt='%.4f')

    print('mAP = '+'%.4f'%mAP+'\tRank-1 = '+'%.4f'%cmc[0])

def compute(query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids):
    # query
    qf = torch.cat(query_feats, dim=0)

    q_pids = np.asarray(query_pids)
    q_camids = np.asarray(query_camids).T

    # gallery
    gf = torch.cat(gallery_feats, dim=0)
    g_pids = np.asarray(gallery_pids)
    g_camids = np.asarray(gallery_camids).T

    m, n = qf.shape[0], gf.shape[0]
    qf = qf.view(m, -1)
    gf = gf.view(n, -1)
    print('Saving feature mat...')
    np.save(args.save_dir+'queryFeat.npy', qf)
    np.save(args.save_dir+'galleryFeat.npy', gf)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    q_camids = np.squeeze(q_camids)
    g_camids = np.squeeze(g_camids)

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP, distmat

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=100):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    max_rank = args.TopK
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print('Saving resulting indexes...', indices.shape)
    np.save(args.save_dir+'result.npy', indices[:, :args.TopK]+1)
    np.savetxt(args.save_dir+'result.txt', indices[:, :args.TopK]+1, fmt='%d')

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

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__=='__main__':
    main()
