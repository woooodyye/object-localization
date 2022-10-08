import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision



# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    boxes, scores = None, None
    #using pytorch's implementation for nms with iou threshold
    #due to lack of time
    indices = torchvision.ops.nms(bounding_boxes, confidence_score, iou_threshold = 0.3)

    return bounding_boxes[indices], confidence_score[indices]


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """

    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates, scores, class_id_to_label):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": float(bbox_coordinates[i][0]),
                "minY": float(bbox_coordinates[i][1]),
                "maxX": float(bbox_coordinates[i][2]),
                "maxY": float(bbox_coordinates[i][3]),
            },
            "class_id": classes[i],
            "scores" : { "score" : float(scores[i])},
            "box_caption": "%s (%.3f)" % (class_id_to_label[classes[i]], scores[i])
        } for i in range(len(classes))
        ]
    return box_list
