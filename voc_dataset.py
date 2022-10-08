''' --------------------------------------------------------
Written by Yufei Ye (https://github.com/JudyYe)
Edited by Anirudh Chakravarthy (https://github.com/anirudh-chakravarthy)
-------------------------------------------------------- '''
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    CLASS_NAMES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    INV_CLASS = {}

    for i in range(len(CLASS_NAMES)):
        INV_CLASS[CLASS_NAMES[i]] = i

    # TODO: Ensure data directory is correct
    def __init__(
        self,
        split='trainval',
        image_size=224,
        top_n=300,
        data_dir='data/VOCdevkit/VOC2007/'
    ):
        super().__init__()
        self.split = split     # 'trainval' or 'test'
        self.data_dir = data_dir
        self.size = image_size
        self.top_n = top_n      # top_n: number of proposals to return

        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.selective_search_dir = os.path.join(
            data_dir, 'selective_search_data')
        self.roi_data = scipy.io.loadmat(
            self.selective_search_dir + '/voc_2007_' + split + '.mat')

        split_file = os.path.join(data_dir, 'ImageSets/Main', split + '.txt')
        with open(split_file) as fp:
            self.index_list = [line.strip() for line in fp]

        self.anno_list = self.preload_anno()

    @classmethod
    def get_class_name(cls, index):
        """
        :return: category name for the corresponding class index.
        """
        return cls.CLASS_NAMES[index]

    @classmethod
    def get_class_index(cls, name):
        """
        :return: class index for the corresponding category name.
        """
        return cls.INV_CLASS[name]

    def __len__(self):
        return len(self.index_list)

    def preload_anno(self):
        """
        :return: a list of labels.
        each element is in the form of [class, weight, gt_class_list, gt_boxes]
         where both class and weight are arrays/tensors in shape of [20],
         gt_class_list is a list of the class ids (separate for each instance)
         gt_boxes is a list of [xmin, ymin, xmax, ymax] values in the range 0 to 1
        """

        # TODO: Make sure you understand how the GT boxes and class labels are loaded
        label_list = []

        for index in self.index_list:
            fpath = os.path.join(self.ann_dir, index + '.xml')
            tree = ET.parse(fpath)
            root = tree.getroot()

            C = np.zeros(20)
            W = np.ones(20) * 2 # default to enable 1 or 0 later for difficulty

            # image h & w to normalize bbox coords
            height = 0
            width = 0

            # new list for each index
            gt_class_list = []
            gt_boxes = []

            for child in root:

                if child.tag == 'size':
                    width = int(child[0].text)
                    height = int(child[1].text)

                if child.tag == 'object':
                    C[self.INV_CLASS[child[0].text]] = 1    # item at index of child name become 1
                    if child[3].text == '1' and W[self.INV_CLASS[child[0].text]] == 2:
                        W[self.INV_CLASS[child[0].text]] = 0    # if not difficult, weight is one
                    elif child[3].text == '0' :
                        W[self.INV_CLASS[child[0].text]] = 1

                    # add class_index to gt_class_list
                    gt_class_list.append(self.INV_CLASS[child[0].text])

                    for t in child:
                        if t.tag == 'bndbox':
                            xmin = int(t[0].text) / width
                            ymin = int(t[1].text) / height
                            xmax = int(t[2].text) / width
                            ymax = int(t[3].text) / height
                            gt_boxes.append([xmin, ymin, xmax, ymax])

            for i in range(len(W)):
                if W[i] == 2:
                    W[i] = 1

            label_list.append([C, W, gt_class_list, gt_boxes])

        return label_list

    def __getitem__(self, index):
        """
        :param index: a int generated by Dataloader in range [0, __len__()]
        :return: index-th element - containing all the aforementioned information
        """

        findex = self.index_list[index]     # findex refers to the file number
        fpath = os.path.join(self.img_dir, findex + '.jpg')

        img = Image.open(fpath)
        width, height = img.size

        img = transforms.ToTensor()(img)
        img = transforms.Resize((self.size, self.size))(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        lab_vec = self.anno_list[index][0]
        wgt_vec = self.anno_list[index][1]

        label = torch.FloatTensor(lab_vec)
        wgt = torch.FloatTensor(wgt_vec)

        gt_class_list, gt_boxes = self.anno_list[index][2], self.anno_list[index][3]

        """
        TODO:
            1. Load bounding box proposals for the index from self.roi_data. The proposals are of the format:
            [y_min, x_min, y_max, x_max] or [top left row, top left col, bottom right row, bottom right col]
            2. Normalize in the range (0, 1) according to image size (be careful of width/height and x/y correspondences)
            3. Make sure to return only the top_n proposals based on proposal confidence ("boxScores")!
            4. You may have to write a custom collate_fn since some of the attributes below may be variable in number for each data point
        """
        proposals = {}

        top_scores = self.roi_data['boxScores'][0][index][0:self.top_n]
        top_boxes = self.roi_data['boxes'][0][index][0:self.top_n]

        normalized_boxes = []

        #normalize bounding boxes
        for box in top_boxes:
            ymin, xmin, ymax, xmax  = box

            xmin = int(xmin) / width
            ymin = int(ymin) / height
            xmax = int(xmax) / width
            ymax = int(ymax) / height
            normalized_boxes.append(torch.FloatTensor([xmin, ymin, xmax, ymax]))


        proposals['top_scores'] = top_scores
        proposals['top_boxes'] = normalized_boxes

        
        ret = {}
        ret['image'] = img
        ret['label'] = label
        ret['wgt'] = wgt
        ret['rois'] = proposals
        ret['gt_boxes'] = gt_boxes
        ret['gt_classes'] = gt_class_list
        return ret



