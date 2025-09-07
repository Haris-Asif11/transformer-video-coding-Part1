#This is basically the same version from test.py
#The aim of this is to pass each image from the val folder to obtain the reconstructed imaeg and prediction and save them in output folder

# This is the code for inference of Detectron2 with ELIC
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
#from testing_custom_class import IntegratedModel
from main import IntegratedModelFinal
import cv2
import torch
import matplotlib.pyplot as plt
from test_utils import *
# Load the configuration file of the trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))


# Set the model weights file
cfg.MODEL.WEIGHTS = ""  # Replace with the actual path to your trained model weights

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model


model = IntegratedModelFinal(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_weights('output_test_17_integrated_model_final/my_integrated_model_weights_final.pth')

def load_state_dict_to_new_model(checkpoint_path, new_model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"]

    # 1) ELIC blocks
    new_model.elic_analysis.load_state_dict({
        k.replace("elic_analysis.", ""): v
        for k, v in state_dict.items() if k.startswith("elic_analysis.")
    }, strict=True)

    new_model.elic_synthesis.load_state_dict({
        k.replace("elic_synthesis.", ""): v
        for k, v in state_dict.items() if k.startswith("elic_synthesis.")
    }, strict=True)

    new_model.hyper_encoder.load_state_dict({
        k.replace("hyper_encoder.", ""): v
        for k, v in state_dict.items() if k.startswith("hyper_encoder.")
    }, strict=True)

    new_model.hyper_decoder.load_state_dict({
        k.replace("hyper_decoder.", ""): v
        for k, v in state_dict.items() if k.startswith("hyper_decoder.")
    }, strict=True)

    # 2) EntropyBottleneck (drop CDF buffers)
    eb_sd = {
        k.replace("entropy_bottleneck.", ""): v
        for k, v in state_dict.items() if k.startswith("entropy_bottleneck.")
    }
    for bad in ("_quantized_cdf", "_cdf_length", "_offset"):
        eb_sd.pop(bad, None)
    new_model.entropy_bottleneck.load_state_dict(eb_sd, strict=False)
    # Rebuild EB tables
    if hasattr(new_model.entropy_bottleneck, "update"):
        new_model.entropy_bottleneck.update(force=True)

    # 3) GaussianConditional (drop CDF buffers as well)
    gc_sd = {
        k.replace("gaussian_conditional.", ""): v
        for k, v in state_dict.items() if k.startswith("gaussian_conditional.")
    }
    for bad in ("_quantized_cdf", "_cdf_length", "_offset", "scale_table"):
        gc_sd.pop(bad, None)
    new_model.gaussian_conditional.load_state_dict(gc_sd, strict=False)

    # Rebuild / set GC tables
    if hasattr(new_model.gaussian_conditional, "update"):
        new_model.gaussian_conditional.update()
    elif hasattr(new_model.gaussian_conditional, "update_scale_table"):
        new_model.gaussian_conditional.update_scale_table(new_model.scale_table)

    print("Components loaded successfully into new model.")

checkpoint_path = 'output_test_31_integrated_model_final/model_0009999.pth'

load_state_dict_to_new_model(checkpoint_path, model)
if hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update"):
    model.gaussian_conditional.update()
elif hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update_scale_table"):
    model.gaussian_conditional.update_scale_table(model.scale_table)

if hasattr(model, 'entropy_bottleneck'):
    model.entropy_bottleneck.update()  # Update Entropy Bottleneck

from detectron2.data import transforms as T




import os
import cv2
import torch
import pickle
import json
import numpy as np


# Define input and output directories
input_dir = '/home/ac35anos/PycharmProjects/detectron2_new/datasets/cityscapes/leftImg8bit/val'
#output_dir = 'output_test25_100k/val'
output_dir = 'output_test31_10k/val'

# Create output directory structure
city_folders = ['frankfurt', 'lindau', 'munster']
for city in city_folders:
    os.makedirs(os.path.join(output_dir, city), exist_ok=True)


def prepare_input(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (2048, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TRAIN)
    #image = transform_gen.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    height = image.shape[1]
    width = image.shape[2]
    input = {'image': image, 'height': height, 'width':width}

    return input


def save_image(tensor, output_path):
    # Convert the tensor to a numpy array
    image_np = tensor.cpu().numpy().astype(np.uint8)

    # Flip the channels from RGB to BGR (if needed)
    image_np = image_np.transpose(1, 2, 0)[:, :, ::-1]

    # Save as PNG
    cv2.imwrite(output_path, image_np)


def save_predictions(predictions, output_path):
    # Save predictions to a JSON file

    # Save predictions to a Pickle file
    with open(output_path + '.pkl', 'wb') as pickle_file:
        pickle.dump(predictions, pickle_file)


# Ensure model is in evaluation mode
model.eval()
total_psnr = 0
total_bpp = 0
total_images = 0

# Iterate through each city folder in the validation directory
for city in city_folders:
    city_input_path = os.path.join(input_dir, city)
    city_output_path = os.path.join(output_dir, city)

    # Get a sorted list of all images in the city folder
    #image_filenames = sorted([f for f in os.listdir(city_input_path) if f.endswith('.png')])
    image_filenames = sorted([f for f in os.listdir(city_input_path)])

    # Iterate through all images in the sorted list
    for image_filename in image_filenames:
        # Prepare input image
        image_path = os.path.join(city_input_path, image_filename)
        if os.path.exists(os.path.join(city_output_path, image_filename)):
            #print(f"Skipping {image_filename} as it is already processed.")
            continue
        input_data = prepare_input(image_path)

        reconstructed_image_name = image_filename
        reconstructed_image_path = os.path.join(city_output_path, reconstructed_image_name)

        # Check if the image and pickle file already exist


        # Run inference on the model
        with torch.no_grad():
            print('Trying to infer: ', image_path)
            reconstructed_image_tensor, predictions, results = model([input_data])
        total_psnr = total_psnr + results['psnr']
        total_bpp = total_bpp + results['bit_rate']
        total_images = total_images + 1


        save_image(reconstructed_image_tensor, reconstructed_image_path)

        '''
        # Save predictions to files
        predictions_output_path = os.path.join(city_output_path,
                                               image_filename.replace('_leftImg8bit.png', '_predictions'))
        save_predictions(predictions[0], predictions_output_path)
        '''

print("Processing completed successfully!")
if total_images != 0:
    print('Average PSNR: ', total_psnr / total_images)
    print('Average BPP: ', total_bpp/total_images)



