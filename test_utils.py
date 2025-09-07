from torchvision import transforms
import torch
import matplotlib.pyplot as plt

def ImagePreprocessForVCT(input_tensor: torch.tensor):
    # Convert uint8 tensor to float and scale values to [0, 1]
    input_tensor = input_tensor.float() / 255.0

    # Normalize the tensor
    # Mean and std need to be for tensors scaled to [0, 1]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = normalize(input_tensor)

    return input_tensor



def ImagePostprocessForVCT(images: torch.Tensor) -> torch.Tensor:

    # Mean and std deviation used during normalization (for tensors scaled to [0, 1])
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    # Ensure the input is on the same device as the mean and std tensors
    mean = mean.to(images.device)
    std = std.to(images.device)

    # Denormalize the images
    images = images * std + mean



    # Scale back from [0, 1] to [0, 255]
    images = images * 255.0

    # Clamp the values to ensure they are valid uint8 values and convert to uint8
    images = torch.clamp(images, 0, 255).byte()

    return images

def insert_images_back_to_list_of_dict(data_list, postprocessed_images):
    """
    Replaces the images in the list of dictionaries with a batch of postprocessed images.

    Parameters:
    - data_list (list of dicts): The original list containing image metadata and tensors from D2's dataloader
    - postprocessed_images (torch.Tensor): The batch of postprocessed images.

    Returns:
    - list of dicts: The updated list with new image tensors.
    """
    # Check if the number of images matches the length of the data list
    if len(data_list) != postprocessed_images.shape[0]:
        raise ValueError("The number of postprocessed images does not match the number of entries in the data list.")

    # Replace each image in the list with the corresponding postprocessed image
    for i, entry in enumerate(data_list):
        entry['image'] = postprocessed_images[i]

    return data_list

def display_tensor_image(tensor):
    # Assuming 'img_tensor' is a PyTorch tensor of shape [C, H, W] and dtype torch.uint8
    if tensor.dim() != 3 or tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be 3D and of dtype torch.uint8")

    # Ensure the tensor is on CPU
    tensor = tensor.cpu()

    # Check the number of channels to determine how to convert it for display
    if tensor.size(0) == 1:
        # For grayscale images, we remove the channel dimension and convert to numpy
        image = tensor.squeeze(0).numpy()
    elif tensor.size(0) == 3:
        # Convert from CxHxW to HxWxC for RGB images and convert to numpy
        image = tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError("Tensor must have 1 or 3 channels")

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show(block=True)

def return_tensor_to_numpy_image(tensor):
    # Assuming 'img_tensor' is a PyTorch tensor of shape [C, H, W] and dtype torch.uint8
    if tensor.dim() != 3 or tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be 3D and of dtype torch.uint8")

    # Ensure the tensor is on CPU
    tensor = tensor.cpu()

    # Check the number of channels to determine how to convert it for display
    if tensor.size(0) == 1:
        # For grayscale images, we remove the channel dimension and convert to numpy
        image = tensor.squeeze(0).numpy()
    elif tensor.size(0) == 3:
        # Convert from CxHxW to HxWxC for RGB images and convert to numpy
        image = tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError("Tensor must have 1 or 3 channels")
    return image

