from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl
import time

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, iou, get_box_data
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int
)
parser.add_argument(
    '--val-interval',
    default=500,
    type=int
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#unfinisehd implementation of maP
def calculate_map():
    """
    Calculate the mAP for classification.
    """
    FP = 0
    TP = 0
    threshold = 0.5
    for class_i in classes:
        for pred_bbox in bboxes:
            #if no gt_box:
            if len(gt_bbox) == 0:
                FP += 1
            # there is gt_box
            else:
                if (mask_gt[index]):
                #valid gt box
                    iou_val = iou(gt_bbox, pred_bbox)
                    if iou_val >= threshold:
                        TP +=1
                        mask_gt[index] = 0
                    else:
                        FP +=1

    


def test_model(model, val_loader=None, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    
    with torch.no_grad():
        
        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            #need to send it to cuda 
            image = data['image'].cuda()
            target = data['label'].cuda()
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            
            rois_bboxes =  torch.squeeze(torch.stack(rois['top_boxes'])).cuda()
            cls_probs = model(image, rois, target) 
            
        # calculate_map()


def train_model(model, train_loader=None, val_loader = None, optimizer=None, args=None, class_id_to_label= None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0

    losses = AverageMeter()

    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            
            rois_bboxes =  torch.squeeze(torch.stack(rois['top_boxes'])).cuda()

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            cls_probs = model(image, rois_bboxes, target) 

            #randomly sample first 10 images to get labeled
            if(iter < 10):
                visualize_bbox(cls_probs, rois_bboxes, image, class_id_to_label)


            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            losses.update(loss.item())
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary

            if iter % args.val_interval == 0 and iter != 0:
                # model.eval()
                # ap = test_model(model, val_loader)
                # print("AP ", ap)
                # model.train()
                wandb.log({'average loss': losses.val})
            

            if iter % args.disp_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch,
                        iter,
                        len(train_loader),
                        loss=losses))

            wandb.log({'loss': loss})
        
    


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    print(args)

    #load dataset
    train_dataset = VOCDataset(split = 'trainval', image_size= 512)
    val_dataset =VOCDataset(split = 'test', image_size= 512)
    class_id_to_label = dict(enumerate(train_dataset.CLASS_NAMES))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    #Free AlexNetlayers #also need to preload alexnet
    for feature in net.features:
        feature.requires_grad = False

    #Create optimizer only for network parameters that are trainable
    params = list(net.parameters())
    
    optimizer = torch.optim.SGD(params[12:], lr=args.lr, 
                            momentum=args.momentum)

    #init wandb
    wandb.init(project="vlr-hw1-q2.5-test")

    # Training
    train_model(net, train_loader, val_loader, optimizer, args, class_id_to_label)


def save_checkpoint(state, is_best, filename='task2_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'task2_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_bbox(cls_probs, rois_bboxes, image, class_id_to_label):
    box_data = []
    for class_num in range(20):
                #get class scores .....
                roi_bbox = rois_bboxes
                roi_scores = cls_probs[:, class_num]
                roi_bbox, roi_scores = filter_threshold(roi_bbox, roi_scores)
                filtered_bbox, filtered_scores = nms(roi_bbox, roi_scores)
                filtered_bbox = filtered_bbox.cpu().detach().numpy()
                filtered_scores = filtered_scores.cpu().detach().numpy()
                box_data = box_data + get_box_data([class_num] * len(filtered_scores), filtered_bbox, filtered_scores
        ,class_id_to_label)
    img = wandb.Image(image, boxes={
    "predictions": {
        "box_data": box_data,
        "class_labels": class_id_to_label,
    },
    })
    wandb.log({"bounding box": img})   

#filter scores with threshold 0.05
def filter_threshold(bbox, scores):
    indices =  (scores >= 0.05).nonzero(as_tuple=True)[0]
    return bbox[indices], torch.flatten(scores[indices])

if __name__ == '__main__':
    main()