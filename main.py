#Final Version
#custom code starts here: merging VCT Encoder/Decoder with Detectron2's Mask RCNN Instance segmentation and bitrate inclusion

from transforms import *
from transforms import ELICAnalysis, ELICSynthesis
from detectron2.checkpoint import DetectionCheckpointer
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import build_model

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
import numpy as np
from test_utils import *

# ---- DEBUG + SAFETY HELPERS -------------------------------------------------


def _ensure_cpu_int_contiguous(x, name):
    if x is None:
        raise RuntimeError(f"{name} is None")
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{name} is not a tensor: {type(x)}")
    x = x.detach().to("cpu", non_blocking=False)  # CompressAI encoders are CPU-side
    if not x.is_contiguous():
        x = x.contiguous()
    # use 32-bit int to be safe for symbols/indexes
    if x.dtype.is_floating_point:
        x = x.round().to(torch.int32)             # symbols should be integer
    elif x.dtype != torch.int32 and x.dtype != torch.int16 and x.dtype != torch.int64:
        x = x.to(torch.int32)
    return x

def _assert_finite(t, name):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaN/Inf] in {name} dtype={t.dtype} shape={tuple(t.shape)}")

def _check_indexes_range(indexes, qcdf, name="indexes"):
    if indexes.numel() == 0:
        raise RuntimeError(f"{name} empty")
    imin = int(indexes.min())
    imax = int(indexes.max())
    if imin < 0:
        raise RuntimeError(f"{name} negative: min={imin}, max={imax}")
    max_allowed = int(qcdf.shape[0]) - 1  # number of scales - 1
    if imax > max_allowed:
        raise RuntimeError(f"{name} out of range: min={imin}, max={imax}, allowed [0,{max_allowed}]")

def safe_gc_compress(gaussian_conditional, y_continuous, indexes, means):
    """
    CompressAI 1.2.1 expects:
      - inputs: continuous y (float), same device as means / module
      - indexes: integer indices (prefer CPU)
      - means: tensor on same device as y
    It internally does quantize(inputs, "symbols", means) -> round(y - means).
    """
    # Buffers must exist (built by .update())
    for nm in ("_quantized_cdf", "_cdf_length", "_offset"):
        if getattr(gaussian_conditional, nm, None) is None:
            raise RuntimeError(f"gaussian_conditional missing {nm}; call gaussian_conditional.update() first.")
    qcdf = gaussian_conditional._quantized_cdf

    # Put y & means on same device (use means' device)
    dev = means.device
    y = y_continuous.detach().to(dev, dtype= torch.float32, non_blocking=False)
    m = means.detach().to(dev, dtype=torch.float32, non_blocking=False)

    # indexes: keep on CPU, int, contiguous
    idx = indexes.detach().to("cpu", dtype=torch.int32, non_blocking=False).contiguous()

    # Basic checks

    if not torch.isfinite(y).all(): raise RuntimeError("[NaN/Inf] in GC y")
    if not torch.isfinite(m).all(): raise RuntimeError("[NaN/Inf] in GC means")
    imin = int(idx.min()); imax = int(idx.max()); max_allowed = int(qcdf.shape[0]) - 1
    if imin < 0 or imax > max_allowed:
        raise RuntimeError(f"GC indexes out of range: min={imin} max={imax} allowed=[0,{max_allowed}]")

    # Call as per 1.2.1 API: compress(inputs, indexes, means=None)
    return gaussian_conditional.compress(y, idx, m)

def safe_eb_compress(entropy_bottleneck, z_continuous):
    """
    EntropyBottleneck.compress expects the continuous latent z (float),
    on the SAME device as the EB buffers (often CUDA). Do NOT cast to int or CPU.
    """
    # Sanity: EB buffers must exist
    if getattr(entropy_bottleneck, "_quantized_cdf", None) is None:
        raise RuntimeError("entropy_bottleneck missing _quantized_cdf; call entropy_bottleneck.update(force=True)")

    # Put z on the same device as EB buffers/parameters
    eb_device = entropy_bottleneck._quantized_cdf.device
    z = z_continuous.detach().to(eb_device, dtype=torch.float32, non_blocking=False)

    # Safety checks
    if not torch.isfinite(z).all():
        raise RuntimeError(f"[NaN/Inf] in EB z symbols: {z.dtype} {tuple(z.shape)}")
    if not z.is_contiguous():
        z = z.contiguous()

    # Call EB.compress (no indexes/means)
    return entropy_bottleneck.compress(z)

def _flat4(shape):
    # Flatten any nesting and ensure exactly 4 ints
    out = []
    def _walk(s):
        if isinstance(s, (list, tuple)):
            for t in s: _walk(t)
        else:
            out.append(int(s))
    _walk(shape)
    out = tuple(out)
    if len(out) != 4:
        raise RuntimeError(f"Bad z_shape for EB.decompress. Need (N,C,H,W), got {out}")
    return out

def safe_eb_decompress(entropy_bottleneck, z_strings, z_shape_4tuple, device):
    """
    Robust EB decode that handles both formats returned by EB.compress:

    Flat (your case):
        z_strings: [bytes] of length N
        -> pass spatial shape only: (H, W)
        -> EB returns (N, C, H, W)

    Nested (per-channel):
        z_strings: [[bytes_for_c] * C] * N
        -> call per batch with (C, H, W), then stack -> (N, C, H, W)
    """
    N, C, H, W = map(int, z_shape_4tuple)

    # Decide structure
    is_nested = (
        isinstance(z_strings, (list, tuple))
        and len(z_strings) > 0
        and isinstance(z_strings[0], (list, tuple))
    )

    # Run EB on CPU to keep VRAM low
    eb_dev = entropy_bottleneck._quantized_cdf.device
    entropy_bottleneck.cpu()

    if not is_nested:
        # Flat list: each element is a single bytes blob for the whole (C,H,W) of that sample.
        # IMPORTANT: pass ONLY (H, W) here; EB will produce (N, C, H, W).
        z_hat_cpu = entropy_bottleneck.decompress(z_strings, (H, W))
        if z_hat_cpu.dim() != 4:
            raise RuntimeError(
                f"EB.decompress(flat) returned {z_hat_cpu.dim()}D tensor with shape {tuple(z_hat_cpu.shape)}"
            )
        if z_hat_cpu.shape[0] != N or z_hat_cpu.shape[1] != C:
            raise RuntimeError(
                f"EB.decompress(flat) shape mismatch. Expect (N,C,H,W)=({N},{C},{H},{W}), "
                f"got {tuple(z_hat_cpu.shape)}"
            )
    else:
        # Nested: per-batch list of per-channel bytes → decode per-batch with (C,H,W)
        if len(z_strings) != N:
            raise RuntimeError(f"EB strings/n mismatch: len={len(z_strings)}, N={N}")
        z_list = []
        for i in range(N):
            if len(z_strings[i]) != C:
                raise RuntimeError(
                    f"EB strings/c mismatch for batch {i}: have {len(z_strings[i])}, expected {C}"
                )
            z_i = entropy_bottleneck.decompress(z_strings[i], (C, H, W))  # -> (C,H,W)
            if z_i.dim() != 3:
                raise RuntimeError(
                    f"EB.decompress(nested) returned {z_i.dim()}D with shape {tuple(z_i.shape)}"
                )
            z_list.append(z_i.unsqueeze(0))  # -> (1,C,H,W)
        z_hat_cpu = torch.cat(z_list, dim=0)  # -> (N,C,H,W)

    # Restore EB device
    entropy_bottleneck.to(eb_dev)

    # Move to requested device
    return z_hat_cpu.to(device, non_blocking=False)



# -----------------------------------------------------------------------------


class HyperpriorEncoder(nn.Module):
    def __init__(self):
        super(HyperpriorEncoder, self).__init__()
        # Simple convolutional network for hyperprior encoder
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class HyperpriorDecoder(nn.Module):
    def __init__(self):
        super(HyperpriorDecoder, self).__init__()
        # Simple convolutional network for hyperprior decoder
        self.conv1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=192 * 2, kernel_size=3, stride=1, padding=1)  # 2x for mean and scale

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x
class IntegratedModelFinal(nn.Module):
    def __init__(self, cfg):
        super(IntegratedModelFinal, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rcnn = build_model(cfg)

        # Freeze RCNN parameters
        for param in self.rcnn.parameters():
            param.requires_grad = False


        self.elic_analysis = ELICAnalysis()
        self.elic_synthesis = ELICSynthesis()
        self.hyper_encoder = HyperpriorEncoder()
        self.hyper_decoder = HyperpriorDecoder()
        self.scale_table = np.exp(np.linspace(-1, 1, 64)).tolist()
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(self.scale_table)
        self.mse_loss = nn.MSELoss()
        self.lambda_rate = 0.1 #decrease for increased image quality but worse rate; original 0.008, (1.3, 23), (1, 30)
        self.alpha_rate = 22

    def quantize(self, x, mode, means=None):
        if mode == "training":
            noise = torch.empty_like(x).uniform_(-0.5, 0.5)
            return x + noise
        elif mode == "dequantize":
            return x - means if means is not None else x
        elif mode == "round":
            return torch.round(x - means) + means if means is not None else torch.round(x)

    def forward(self, inputs):
        # Compress and decompress the images
        '''
        batched_inputs = []
        for image, target in zip(images, targets):
            # Create a dictionary for each image
            img_dict = {
                "image": image,  # This should be a torch tensor of the image data
                "instances": target  # Assuming 'target' is an Instances object or similar
            }
            batched_inputs.append(img_dict)
        '''
        if self.training:

            #print('device: ', self.device)
            # Extract image tensors from the list of dictionaries
            image_tensors = [entry['image'] for entry in inputs]

            # Stack all image tensors to form a batch
            batch_tensors_VCT = torch.stack(image_tensors, dim=0)

            batch_tensors_VCT_preprocessed = ImagePreprocessForVCT(batch_tensors_VCT)

            # ELIC block expects tensors of shape [B, T, C, H, W]
            batch_tensors_VCT_preprocessed = batch_tensors_VCT_preprocessed.unsqueeze(1)

            batch_tensors_VCT_preprocessed = batch_tensors_VCT_preprocessed.to(self.device)

            y = self.elic_analysis(batch_tensors_VCT_preprocessed)
            y = y.squeeze(1)

            #----------------------------------------------------------------------------------------
            #following added for testing
            y_q = y
            #----------------------------------------------------------------------------------------
            z = self.hyper_encoder(y_q)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.hyper_decoder(z_hat)
            means, scales = params.chunk(2, 1)  # Split the params into means and scales
            y_hat, y_likelihoods = self.gaussian_conditional(y_q, scales, means)
            y_hat = self.quantize(y_hat, mode="dequantize", means=means)
            y_hat = y_hat.unsqueeze(1)
            x_hat = self.elic_synthesis(y_hat, batch_tensors_VCT_preprocessed.shape)

            epsilon = 1e-9
            rate_y = torch.sum(-torch.log(y_likelihoods + epsilon))
            rate_z = torch.sum(-torch.log(z_likelihoods + epsilon))
            rate = rate_y + rate_z
            average_rate = rate / (2 * 512 * 1024)  # hard coded for now; assumes 2 images in batch
            lambda_average_rate = self.lambda_rate * average_rate
            distortion= self.mse_loss(x_hat, batch_tensors_VCT_preprocessed) #x_hat represents the reconstructed images
            alpha_distortion = self.alpha_rate * distortion


            batch_tensors_VCT_postprocessed = ImagePostprocessForVCT(x_hat)
            batch_tensors_VCT_postprocessed = batch_tensors_VCT_postprocessed.squeeze(1)


            inputs = insert_images_back_to_list_of_dict(inputs, batch_tensors_VCT_postprocessed)


            outputs = self.rcnn(inputs)  # need decompressed here instead of images
            outputs['mse_loss'] = alpha_distortion #Adding weight to prioritizing ELIC's loss criterion
            outputs['lambda_average_rate'] = lambda_average_rate
            # Add the mse_loss to the losses dict from RCNN
            # print('outputs: ------------------------------', outputs)
            return outputs
        else:
            #Assume prediction on batch of 1 image only
            with torch.no_grad():

                #following line till 275 added for testing
                compression_output = self.compress(inputs)

                self.last_compress_info = {
                    'z_shape': tuple(compression_output['z_shape']),
                    'x_shape': tuple(compression_output['x_shape']),
                }

                total_bits = self.calculate_total_bits(compression_output)
                print("Total bits:", total_bits)
                bpp = total_bits / (1024 * 2048)
                print("BPP:", bpp)

                x_hat = self.decompress(compression_output['strings'],
                                        compression_output['y_shape'],
                                        compression_output['x_shape'])


                batch_tensor_VCT_postprocessed = ImagePostprocessForVCT(x_hat)

                # Change image tensor back to [C, H, W]
                batch_tensor_VCT_postprocessed = batch_tensor_VCT_postprocessed.squeeze()


                psnr_value = self.calculate_psnr(inputs[0]['image'], batch_tensor_VCT_postprocessed)
                print(f'PSNR: {psnr_value} DB')
                


                inputs[0]['image'] = batch_tensor_VCT_postprocessed

                results = {'psnr': psnr_value,
                           'bit_rate': bpp }

                return batch_tensor_VCT_postprocessed, self.rcnn(inputs), results


    def compress(self, inputs):
        image_tensor = inputs[0]['image']
        batch_tensor = ImagePreprocessForVCT(image_tensor)
        batch_tensor = batch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Analysis
        y = self.elic_analysis(batch_tensor).squeeze(1)  # [1,192,h,w]

        # Hyperprior
        z = self.hyper_encoder(y)  # [1,192,hz,wz]

        # 1) Get quantized z for params via forward (no big allocs)
        z_hat, _ = self.entropy_bottleneck(z)

        # 2) Get bitstream strings for z (no need to decompress now)
        z_strings = safe_eb_compress(self.entropy_bottleneck, z)
        print("EB strings type:", type(z_strings), "len:", len(z_strings))
        if len(z_strings) > 0:
            print("EB strings[0] type:", type(z_strings[0]),
                  "len0:", (len(z_strings[0]) if hasattr(z_strings[0], '__len__') else 'n/a'))

        # Decode hyperprior to get means/scales
        params = self.hyper_decoder(z_hat)
        means, scales = params.chunk(2, 1)
        scales = scales.clamp_min(1e-9)

        # Build GC indexes and symbols
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = safe_gc_compress(self.gaussian_conditional, y, indexes, means)
        # Save shapes needed for decode (note: EB needs z_shape)

        return {
            'strings': [y_strings, z_strings],
            'y_shape': tuple(y.shape),  # (N, C, H, W)
            'z_shape': tuple(z.shape),  # (N, C, H, W)
            'x_shape': tuple(batch_tensor.shape)  # (N, T, C, H, W)
        }


    def decompress(self, strings, shape, x_shape):
        y_strings, z_strings = strings

        # normalize shape (you already set self.last_compress_info earlier)
        z_shape = tuple(self.last_compress_info['z_shape'])
        dev = next(self.parameters()).device

        # FIXED: robust EB decode
        z_hat = safe_eb_decompress(self.entropy_bottleneck, z_strings, z_shape, dev)

        params = self.hyper_decoder(z_hat)
        means, scales = params.chunk(2, 1)
        scales = scales.clamp_min(1e-9)

        # GC on CPU to keep VRAM low
        indexes = self.gaussian_conditional.build_indexes(scales)
        idx_cpu = indexes.detach().to("cpu", dtype=torch.int32, non_blocking=False).contiguous()
        y_hat_cpu = self.gaussian_conditional.decompress(y_strings, idx_cpu, dtype=torch.float32)

        y_hat = y_hat_cpu.to(dev, non_blocking=False)
        y_hat = self.quantize(y_hat, mode="dequantize", means=means)

        y_hat = y_hat.unsqueeze(1)
        x_hat = self.elic_synthesis(y_hat, self.last_compress_info['x_shape']).squeeze(1)
        return x_hat

    def calculate_total_bits(self, compressed_data):
        total_bytes = 0
        for data_list in compressed_data['strings']:
            for data in data_list:
                total_bytes += len(data)  # Add the length of each byte string
        total_bits = total_bytes * 8  # Convert total bytes to bits
        return total_bits

    def calculate_psnr(self, original_img, decompressed_img, max_pixel=255):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_img = original_img.to(device)
        mse = torch.mean((original_img - decompressed_img) ** 2)
        if mse == 0:  # This means no noise is present in the signal.
            return float('inf')
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))

    def load_weights_rcnn(self, weight_path):
        """
        Load weights from a specified file into the RCNN model of this model instance.

        Parameters:
        weight_path (str): Path to the weights file (.pth or .pt format).
        """

        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

        filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.rcnn.state_dict()}
        self.rcnn.load_state_dict(filtered_state_dict, strict=False)
        print("Weights loaded successfully into the RCNN model.")


    def load_weights_elic_hyper_gaussian_entropy_bottleneck(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']

        # Print keys for ELIC Analysis
        #elic_analysis_keys = [k for k in state_dict.keys() if k.startswith('elic_analysis.')]
        #print("Keys for ELIC Analysis:", elic_analysis_keys)

        # Load state dict selectively
        self.elic_analysis.load_state_dict(
            {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')})
        self.elic_synthesis.load_state_dict(
            {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')})
        self.hyper_encoder.load_state_dict(
            {k.replace('hyper_encoder.', ''): v for k, v in state_dict.items() if k.startswith('hyper_encoder.')})
        self.hyper_decoder.load_state_dict(
            {k.replace('hyper_decoder.', ''): v for k, v in state_dict.items() if k.startswith('hyper_decoder.')})
        self.entropy_bottleneck.load_state_dict(
            {k.replace('entropy_bottleneck.', ''): v for k, v in state_dict.items() if
             k.startswith('entropy_bottleneck.')})
        self.gaussian_conditional.load_state_dict(
            {k.replace('gaussian_conditional.', ''): v for k, v in state_dict.items() if
             k.startswith('gaussian_conditional.')})

        # >>> STEP 2 safeguard: rebuild probability tables after load <<<
        if hasattr(self, "gaussian_conditional") and hasattr(self.gaussian_conditional, "update"):
            self.gaussian_conditional.update()
        elif hasattr(self, "gaussian_conditional") and hasattr(self.gaussian_conditional, "update_scale_table"):
            self.gaussian_conditional.update_scale_table(self.scale_table)

        if hasattr(self, "entropy_bottleneck"):
            self.entropy_bottleneck.update(force=True)
        # <<< STEP 2 end >>>


        print("Components (ELIC blocks, Hyper encoder and decoder, entropy bottleneck and gaussian conditional) loaded successfully into new model.")


    def load_elic_weights(self, analysis_state_dict, synthesis_state_dict):
        """
        Load weights into ELIC analysis and synthesis blocks from specified files.

        Parameters:
        analysis_weight_path (str): Path to the analysis block weights (.pth or .pt format).
        synthesis_weight_path (str): Path to the synthesis block weights (.pth or .pt format).
        """
        # Load analysis weights
        #analysis_state_dict = torch.load(analysis_weight_path, map_location=torch.device('cpu'))
        self.elic_analysis.load_state_dict(analysis_state_dict)
        print("Weights loaded successfully into the ELIC analysis block.")

        # Load synthesis weights
        #synthesis_state_dict = torch.load(synthesis_weight_path, map_location=torch.device('cpu'))
        self.elic_synthesis.load_state_dict(synthesis_state_dict)
        print("Weights loaded successfully into the ELIC synthesis block.")

    def save_model_weights(self, filename):
        """
        Saves the weights of a given PyTorch model.

        Args:
        model (torch.nn.Module): The model whose weights are to be saved.
        filename (str): The path to the file where the model weights will be saved.
        """
        # Save the model state dictionary
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        # Load the original state dictionary from the file
        original_state_dict = torch.load(filename, map_location=self.device)
        # Adjust the keys by adding 'rcnn.' prefix
        #adjusted_state_dict = {'rcnn.' + key: value for key, value in original_state_dict.items()}
        # Load the adjusted state dictionary into the model
        self.load_state_dict(original_state_dict)






from detectron2.engine import DefaultTrainer
import time
from detectron2.utils import comm
from test_utils import ImagePreprocessForVCT

class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)  # Ensure the superclass's init is called if necessary
        # Explicitly setting self.model


    @classmethod
    def build_model(cls, cfg):
        """
        Return an instance of the IntegratedModel.
        """
        model = IntegratedModelFinal(cfg)
        model.load_weights_rcnn("scratch_pad_model_final.pth")
        #load pretrained weights for the ELIC components
        model.load_weights_elic_hyper_gaussian_entropy_bottleneck('Compressor_w_Entropy_Model_w_ELIC_test2_w_schedule_and_gradientclip_lambda0.08_epochs20_512.pth')

        # >>> rebuild probability tables <<<
        # GaussianConditional: no args
        if hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update"):
            model.gaussian_conditional.update()
        elif hasattr(model, "gaussian_conditional") and hasattr(model.gaussian_conditional, "update_scale_table"):
            model.gaussian_conditional.update_scale_table(model.scale_table)

        # EntropyBottleneck: force=True is valid
        if hasattr(model, "entropy_bottleneck"):
            model.entropy_bottleneck.update(force=True)
        # <<< end rebuild >>>
        #model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model







# Assuming IntegratedModel is as defined in your training code
from detectron2.config import get_cfg
from detectron2 import model_zoo
# Configuration as per your training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))




#training code begins here
import pickle

#path1 = 'train_dataset_cityscape.pickle'
#path2 = 'val_dataset_cityscape.pickle'

path1 = 'pickled_dataset_files/train_dataset_cityscape_3.pickle'
path2 = 'pickled_dataset_files/val_dataset_cityscape_3.pickle'

with open(path1, 'rb') as file:
    train_loader = pickle.load(file)

with open(path2, 'rb') as file:
    val_loader = pickle.load(file)


# training from scratch
import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg


# Register Cityscapes dataset
if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

    cfg.MODEL.WEIGHTS = ''
    # Set training options
    # cfg.DATASETS.TRAIN = ("cityscapes_train",)
    # cfg.DATASETS.TEST = ("cityscapes_val",)
    cfg.DATALOADER.TRAIN = (train_loader,)
    cfg.DATALOADER.TEST = (val_loader,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001 # original 0.00025
    cfg.SOLVER.MAX_ITER = 60000 #60k
    cfg.SOLVER.STEPS = (1500, 3000, 4500, 6000, 7500, 9000)
    #cfg.SOLVER.OPTIMIZER = "Adam"
    #cfg.SOLVER.STEPS = (500, 700)
    cfg.SOLVER.GAMMA = 0.92
    cfg.SOLVER.WARMUP_ITERS = 0

    # Set model output directory
    cfg.OUTPUT_DIR = "outputs/output_test_33_integrated_model_final"

    # Set other options such as seed, evaluation, etc.
    cfg.SEED = 42
    cfg.TEST.EVAL_PERIOD = 100

    # Create a trainer and train the model
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

