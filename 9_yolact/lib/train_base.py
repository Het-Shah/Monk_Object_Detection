import os
import sys
import subprocess

sys.path.append("yolact")

import ast
import pprint

# import yolact
from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact

system_dict = {}



#######################################################################################################################################
def set_dataset_params(root_dir="data", coco_dir="coco", imageset="traincustom", annotation_path="annotations", class_names=None):
    system_dict["dataset_root"] = root_dir
    system_dict["dataset"] = coco_dir
    system_dict["dataset_dir"] = root_dir + "/" + coco_dir
    system_dict["imageset"] = imageset
    system_dict["annotation_dir"] = annotation_path
    system_dict["class_names"] = class_names


def set_img_preproc_params(img_short_side=600, img_long_side=1000, mean=MEANS, std=STD):
    system_dict["img_short_side"] = img_short_side
    system_dict["img_long_side"] = img_long_side
    system_dict["img_pixel_means"] = str(mean)
    system_dict["img_pixel_stds"] = str(std)
    system_dict["img_pixel_means"] = ast.literal_eval(system_dict["img_pixel_means"])
    system_dict["img_pixel_stds"] = ast.literal_eval(system_dict["img_pixel_stds"])

def set_hyper_params(gpus=[0], lr=0.001, lr_decay_epoch="7", epochs=10, batch_size=1):
    if gpus == '0':
        system_dict["gpus"] = list(gpus)
    else :
        system_dict["gpus"] = gpus
    system_dict["lr"] = lr
    system_dict["lr_decay_epoch"] = lr_decay_epoch
    system_dict["epochs"] = epochs
    system_dict["rcnn_batch_size"] = batch_size

def set_output_params(log_interval=100, save_prefix="model_vgg16"):
    system_dict["log_interval"] = log_interval
    if(not os.path.isdir("trained_model")):
        os.mkdir("trained_model")
    system_dict["save_prefix"] = "trained_model/" + save_prefix


anno = ['kangaroo']
set_dataset_params(root_dir="/home/het/Monk_Object_Detection/example_notebooks/sample_dataset/kangaroo", coco_dir="",imageset="Images",annotation_path="annotations/instances_Images.json", class_names=anno)

# cust_dataset and cust_yolact_base_config functions written in data/config.py
dataset = cust_dataset(name='Custom', 
                       train_images=os.path.join(system_dict['dataset_dir'],system_dict['imageset']), 
                       train_info=os.path.join(system_dict['dataset_dir'],system_dict['annotation_dir']), 
                       valid_images=os.path.join(system_dict['dataset_dir'],system_dict['imageset']), 
                       valid_info=os.path.join(system_dict['dataset_dir'],system_dict['annotation_dir']), 
                       has_gt=True, 
                       class_names=system_dict["class_names"], 
                       label_map=None)

yolact_base_config = cust_yolact_base_config(name="cust_base",
                                             dataset=dataset,
                                             max_size=550)

cfg = yolact_base_config.copy()

# print(yolact_base_config.dataset.class_names)
dataset = COCODetection(image_path=cfg.dataset.train_images,
                        info_file=cfg.dataset.train_info,
                        transform=SSDAugmentation(MEANS))


print(dataset)