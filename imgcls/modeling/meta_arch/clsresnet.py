import torch
import torch.nn as nn
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone, resnet


@META_ARCH_REGISTRY.register()
class ClsResNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.RESNETS.NUM_CLASSES
        self.in_features = cfg.MODEL.RESNETS.IN_FEATURES
        self.bottom_up = build_backbone(cfg)
        self.criterion = nn.CrossEntropyLoss()
        
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)  # Do not need size_divisibility
        return images

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_labels = [x['label'] for x in batched_inputs]
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long).to(self.device)
        features = self.bottom_up(images.tensor)
        features = [features[f] for f in self.in_features]
        
        if self.training:
            losses = self.losses(gt_labels, features)
            return losses
        else:
            results = self.inferece(features)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                processed_results.append({"pred_classes": results_per_image})
            return processed_results

    def forward_imgnet(self, images):
        features = self.bottom_up(images)
        return features["linear"]


    def inferece(self, features, topk=1):
        _, pred = features[0].topk(topk, 1, True, True)
        return pred
        

    def losses(self, gt_labels, features):
        return {"loss_cls": self.criterion(features[0], gt_labels)}

