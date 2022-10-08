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
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace = True),
            nn.Linear(in_features=4096, out_features= 4096),
            nn.ReLU(inplace = True)
        )

        self.score_fc = nn.Linear(in_features=4096, out_features=20)
        self.bbox_fc = nn.Linear(in_features=4096, out_features=20)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):


        # compute cls_prob which are N_roi X 20 scores
         #calculate image features
        image_features = self.features(image)
       
        #makesure rois are abosolute values
        rois_abs = torch.abs(rois)

        #pool results and reshape #spatial resolution is 31 bc the output shape for 
        #features is 1x256x31x31, we normalized our rois, so we need to scale up
        pooled_features = self.roi_pool(image_features, [rois_abs], output_size=(6,6), spatial_scale= 31) # need to write down spatial scale and output size 
        pooled_features = pooled_features.view(pooled_features.size()[0], -1)
        #pass into classifier
        classified_features = self.classifier(pooled_features)

        #get score and bbox
        score_det = self.score_fc(classified_features)
        bbox_det = self.bbox_fc(classified_features)
        
        #apply softmax across differnt dimensions
        score_softmax = F.softmax(score_det, dim = 1)
        bbox_softmax = F.softmax(bbox_det, dim = 0)

        #do hadarmard
        cls_prob = score_softmax * bbox_softmax

        if self.training:
            label_vec = gt_vec.view(-1,self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        cls_sum = torch.sum(cls_prob, 0).view(-1, self.n_classes)
        loss = nn.BCELoss(reduction = "sum")(torch.clamp(cls_sum, min=0.0, max=1.0), label_vec)

        return loss
