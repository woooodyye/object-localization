import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2) ),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace = True)
        )
        self.roi_pool = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.ReLU(inplace = True),
            nn.Linear(),
            nn.ReLU(inplace = True)
        )

        self.score_fc = nn.Linear()
        self.bbox_fc = nn.Linear()

        # loss
        self.cross_entropy = nn.BCELoss()

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):


        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        image_features = self.features(image)
        rois_abs = torch.abs(rois.data)
        pooled_features = self.roi_pool(image, rois_abs) # need to write down spatial scale and output size 

        score_det = self.score_fc(pooled_features)
        bbox_det = self.bbox_fc(pooled_features)

        score_softmax = F.softmax(score_det)
        bbox_softmax = F.softmax(bbox_det)

        combined_score = score_softmax * bbox_softmax

        #do a sum over some dimension
        cls_prob = torch.sum(combined_score, 0).view(-1, self.n_classes)

        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        loss = F.binary_cross_entropy(cls_prob, label_vec)

        return loss
