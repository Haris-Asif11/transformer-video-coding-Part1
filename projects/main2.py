'''
import sys, os, distutils.core

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random




# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = cv2.imread('cars_test.jpg')
im = cv2.resize(im, (640, 480))
print(im.shape)
#cv2.imshow('test image', im)
#cv2.waitKey(0)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs['instances'].pred_classes)
print(outputs['instances'].pred_boxes)

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('Model Output Visualised', out.get_image()[:, :, ::-1])
cv2.waitKey(0)

'''


'''

#old working code starts here
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Load the configuration file of the trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

# Set the model weights file
cfg.MODEL.WEIGHTS = "output/model_0009999.pth"  # Replace with the actual path to your trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model

# Create a predictor using the trained model
predictor = DefaultPredictor(cfg)

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the custom image

image = cv2.imread('cars_test.jpg')
#image = cv2.resize(image, (640, 480))

# Make predictions
outputs = predictor(image)

# Visualization
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image())
plt.show()

'''

'''
#using detectrons own scrips to load train and test sets for training

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import pickle


def load_instance_datasets(CfgMaskRcnn):
    from detectron2.data.datasets.builtin import register_all_cityscapes
    from detectron2.data import build_detection_test_loader, build_detection_train_loader
    mapper = None
    dataset = build_detection_train_loader(CfgMaskRcnn, None)  # "custom_dataset")
    dataset_val = build_detection_test_loader(CfgMaskRcnn, "cityscapes_fine_instance_seg_val")
    return dataset, dataset_val

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

train_dataset, val_dataset = load_instance_datasets(cfg) #val is the test set here

path1 = 'train_dataset_cityscape.pickle'
path2 = 'val_dataset_cityscape.pickle'
with open(path1, 'wb') as file:
    pickle.dump(train_dataset, file)

with open(path2, 'wb') as file:
    pickle.dump(val_dataset, file)

'''

'''
#for displaying image in dataloader
#import pickle

path1 = 'train_dataset_cityscape.pickle'
path2 = 'val_dataset_cityscape.pickle'

with open(path1, 'rb') as file:
    train_loader = pickle.load(file)

with open(path2, 'rb') as file:
    val_loader = pickle.load(file)

for ind, batch in enumerate(train_loader):
    print(batch[0]['image'].shape)
    # images, ann = batch[0]['image'], batch[0]['instances']
    # print('Image batch shape: ', images.shape)
    
'''



'''
#training from scratch
import os
import torch, detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator



# Register Cityscapes dataset

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

# Set training options
#cfg.DATASETS.TRAIN = ("cityscapes_train",)
#cfg.DATASETS.TEST = ("cityscapes_val",)
cfg.DATALOADER.TRAIN = (train_loader, )
cfg.DATALOADER.TEST = (val_loader, )
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = (500, 800)
cfg.SOLVER.GAMMA = 0.05

# Set model output directory
cfg.OUTPUT_DIR = "output"

# Set other options such as seed, evaluation, etc.
cfg.SEED = 42
cfg.TEST.EVAL_PERIOD = 100

# Create a trainer and train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Optionally, evaluate the trained model
evaluator = COCOEvaluator("cityscapes_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg, trainer.model, evaluators=[evaluator])
'''

#custom code starts here: merging VCT Encoder/Decoder with detectron2's Mask RCNN Instance segmentation

from transforms import *
from transforms import ELICAnalysis, ELICSynthesis

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import build_model

class IntegratedModel(nn.Module):
    def __init__(self, cfg):
        super(IntegratedModel, self).__init__()
        self.rcnn = build_model(cfg)
        self.elic_analysis = ELICAnalysis()
        self.elic_synthesis = ELICSynthesis()

    def forward(self, images, targets=None):
        # Compress and decompress the images
        compressed = self.elic_analysis(images)
        decompressed = self.elic_synthesis(compressed)

        # Compute the MSE loss between the original and decompressed images
        mse_loss = F.mse_loss(decompressed, images)

        # If targets are provided, compute detection losses as well
        if targets is not None:
            outputs = self.rcnn(decompressed, targets)
            # Add the mse_loss to the losses dict from RCNN
            outputs["losses"]["mse_loss"] = mse_loss
            return outputs
        else:
            return decompressed

from detectron2.engine import DefaultTrainer

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        """
        Return an instance of the IntegratedModel.
        """
        model = IntegratedModel(cfg)  # This ensures your model includes the ELIC blocks
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Create the optimizer and include parameters from both RCNN and ELIC blocks.
        """
        # You need to ensure that the model passed here is your IntegratedModel
        # and that it has attributes rcnn, elic_analysis, and elic_synthesis properly defined.

        # Parameters from the detectron2 RCNN model
        rcnn_params = model.rcnn.parameters()
        # Parameters from the ELICAnalysis block
        elic_analysis_params = model.elic_analysis.parameters()
        # Parameters from the ELICSynthesis block
        elic_synthesis_params = model.elic_synthesis.parameters()

        # Combine all parameters into a list; you might want to use different settings for different parts
        params = [
            {"params": rcnn_params, "lr": cfg.SOLVER.BASE_LR},  # Standard learning rate for RCNN
            {"params": elic_analysis_params, "lr": cfg.SOLVER.BASE_LR * 0.1},
            # Optionally, a different learning rate for ELIC blocks
            {"params": elic_synthesis_params, "lr": cfg.SOLVER.BASE_LR * 0.1}
        ]

        # Create the optimizer with these combined parameters
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
        return optimizer

    @classmethod
    def run_step(self):
        """
        Implement a training step with your custom loss handling.
        """
        assert self.model.training, "[CustomTrainer] model was changed to eval mode!"
        inputs = next(self._data_loader_iter)
        outputs = self.model(inputs["images"], inputs["targets"])

        # Sum all losses (including the added MSE loss)
        losses = sum(loss for loss in outputs["losses"].values())
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()



import pickle

path1 = 'train_dataset_cityscape.pickle'
path2 = 'val_dataset_cityscape.pickle'

with open(path1, 'rb') as file:
    train_loader = pickle.load(file)

with open(path2, 'rb') as file:
    val_loader = pickle.load(file)


# training from scratch
import os
import torch, detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

# Register Cityscapes dataset

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

# Set training options
# cfg.DATASETS.TRAIN = ("cityscapes_train",)
# cfg.DATASETS.TEST = ("cityscapes_val",)
cfg.DATALOADER.TRAIN = (train_loader,)
cfg.DATALOADER.TEST = (val_loader,)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = (500, 800)
cfg.SOLVER.GAMMA = 0.05

# Set model output directory
cfg.OUTPUT_DIR = "output_test"

# Set other options such as seed, evaluation, etc.
cfg.SEED = 42
cfg.TEST.EVAL_PERIOD = 100

# Create a trainer and train the model
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()