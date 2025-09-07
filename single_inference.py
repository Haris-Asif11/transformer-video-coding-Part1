import os
# Kill thread races (OpenMP/MKL/OpenCV):
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force useful traces and predictable kernels:
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # even if you end up on CUDA later
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["PYTORCH_JIT"] = "0"

import faulthandler; faulthandler.enable()  # prints native backtrace on segfault

import cv2
cv2.setNumThreads(0)

import torch
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False
torch.backends.mkldnn.enabled = False


# This is the code for inference of Detectron2 with ELIC
from detectron2.config import get_cfg
from detectron2 import model_zoo
#from testing_custom_class import IntegratedModel
from main import IntegratedModelFinal
import cv2

from test_utils import *

# Load the configuration file of the trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))


# Set the model weights file
cfg.MODEL.WEIGHTS = ""  # Replace with the actual path to your trained model weights
#cfg.MODEL.WEIGHTS = "output_original_testing_2/model_final.pth"  # Replace with the actual path to your trained model weights
#cfg.MODEL.WEIGHTS = 'output_original_testing_version2/model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model

model = IntegratedModelFinal(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#The following line actually loads the mask rcnn weights needed for segmentation, the rest are overwritten
model.load_weights('outputs/output_test_17_integrated_model_final/my_integrated_model_weights_final.pth')



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


checkpoint_path = 'outputs/output_test_29_integrated_model_final/model_0059999.pth'
load_state_dict_to_new_model(checkpoint_path, model)


if hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update"):
    model.gaussian_conditional.update()
elif hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update_scale_table"):
    model.gaussian_conditional.update_scale_table(model.scale_table)

if hasattr(model, 'entropy_bottleneck'):
    model.entropy_bottleneck.update()  # Update Entropy Bottleneck





model.eval()

from detectron2.data import transforms as T

def prepare_input(image_path):
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (2000, 1000))
    image = cv2.resize(image, (2048, 1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TRAIN)
    #image = transform_gen.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    height = image.shape[1]
    width = image.shape[2]
    input = {'image': image, 'height': height, 'width':width}

    return input




image_path = '/home/haris/PycharmProjects/Detectron2_new/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png'
image = cv2.imread(image_path)

input = prepare_input(image_path)


with torch.no_grad():
    # Assuming the model expects a batch of images, even if you're just passing one
    reconstructed_image_tensor, predictions, _ = model([input])

display_tensor_image(reconstructed_image_tensor)
im = return_tensor_to_numpy_image(reconstructed_image_tensor)

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

#print(predictions[0])
# Assuming predictions are formatted similarly to Detectron2 outputs
v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.6)
out = v.draw_instance_predictions(predictions[0]["instances"].to("cpu"))
cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()


