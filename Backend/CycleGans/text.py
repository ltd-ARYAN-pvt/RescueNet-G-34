import torch
from PIL import Image
from generator import Generator
import numpy as np
import config

# Load the trained generator weights
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
state_dict=torch.load(config.CHECKPOINT_GEN_Z)["state_dict"]
gen_Z.load_state_dict(state_dict)
gen_Z.eval()

def load_image(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmentations = config.transforms(image=image, image0=image)
    image = augmentations["image"]
    return image.to(config.DEVICE)

# Test the generator on a sample image
image_path = '01_hazy.png'
input_image = load_image(image_path)
with torch.no_grad():
    output_image = gen_Z(input_image)

# Save the output image
output_image = (output_image + 1) / 2
output_image = output_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
output_image = (output_image * 255).astype(np.uint8)
Image.fromarray(output_image).save('output.jpg')