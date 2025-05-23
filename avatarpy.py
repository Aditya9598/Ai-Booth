# -*- coding: utf-8 -*-
"""avatar.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1torCwxKU9GuWcoS6E07-QnQk8pjvF87Z
"""

!pip install -q diffusers transformers accelerate torch gradio
!pip install -q xformers  # Optional for faster inference

import os
from google.colab import files
import shutil
from IPython.display import clear_output

# Create folder for user images
os.makedirs("user_images", exist_ok=True)

# Initialize counter
upload_count = 0

# Clear existing files in directory
for f in os.listdir("user_images"):
    os.remove(os.path.join("user_images", f))

while True:
    try:
        # Upload images
        clear_output()
        print(f"Upload exactly 5 images ({upload_count}/5 uploaded)")
        uploaded = files.upload()

        # Validate count
        if len(uploaded) + upload_count > 5:
            print(f"Error: You uploaded {len(uploaded)} images. Total should be exactly 5.")
            continue

        # Validate file types
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        for filename in uploaded.keys():
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file type: {filename}. Only images allowed (jpg, png, webp)")

        # Move valid files
        for filename in uploaded.keys():
            shutil.move(filename, os.path.join("user_images", filename))

        upload_count += len(uploaded)

        if upload_count == 5:
            print("Success! 5 images uploaded")
            break

    except Exception as e:
        print(f"Error: {str(e)}")
        # Reset on error
        for f in uploaded.keys():
            if os.path.exists(f):
                os.remove(f)
        upload_count = 0
        continue

print("\nUploaded images:")
!ls user_images

from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Configure DreamBooth training
pipe.unet.train()
pipe.text_encoder.train()

# Training settings (adjust based on your needs)
training_steps = 800
learning_rate = 2e-6

# Train on user images (simplified example)
# Note: For full DreamBooth, use a script like https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
for step in range(training_steps):
    # Load a batch of user images
    # Add your training loop here (see Hugging Face DreamBooth example for details)
    pass

# Save the fine-tuned model
pipe.save_pretrained("fine_tuned_superman")

# inference and seed

# Google Colab Notebook for Face Blending Enhancement in Avatar Generation

# Install necessary dependencies
!pip install diffusers transformers accelerate safetensors
!pip install git+https://github.com/TencentARC/GFPGAN.git

# Import libraries
import torch
from diffusers import StableDiffusionPipeline
from gfpgan import GFPGANer
from PIL import Image, ImageDraw
import os
import numpy as np

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

# Load Stable Diffusion (or SDXL) pipeline with FP16 for better performance
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe.to(device)
except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    pipe = None

# Load GFPGAN for face restoration with a specific model path
try:
    model_path = "GFPGANv1.4.pth"
    if not os.path.exists(model_path):
        os.system(f"wget https://github.com/TencentARC/GFPGAN/releases/download/v1.4/{model_path}")

    restorer = GFPGANer(model_path=model_path, upscale=2, arch='clean', channel_multiplier=2)
except Exception as e:
    print(f"Error loading GFPGAN model: {e}")
    restorer = None


def generate_avatar(prompt, num_images=1):
    """Generate an avatar from a text prompt."""
    if pipe is None:
        print("Stable Diffusion model is not loaded.")
        return None

    try:
        images = pipe(prompt, num_inference_steps=50, num_images_per_prompt=num_images).images
        return images
    except Exception as e:
        print(f"Error generating avatars: {e}")
        return None


def enhance_face(image):
    """Enhance face blending using GFPGAN."""
    if restorer is None:
        print("GFPGAN model is not loaded.")
        return image

    try:
        image_np = np.array(image)
        _, _, restored_image = restorer.enhance(image_np, has_aligned=False, only_center_face=False, paste_back=True)
        return Image.fromarray(restored_image)
    except Exception as e:
        print(f"Error enhancing face: {e}")
        return image


def save_images(images, prefix="avatar"):
    """Save generated images to disk."""
    for i, img in enumerate(images):
        img_path = os.path.join("outputs", f"{prefix}_{i+1}.png")
        img.save(img_path)
        print(f"Saved: {img_path}")


# Example usage
prompt = "A futuristic cyberpunk warrior with a neon mask"
generated_avatars = generate_avatar(prompt, num_images=2)

if generated_avatars:
    enhanced_avatars = [enhance_face(img) for img in generated_avatars]

    # Display results
    for i, (gen_img, enh_img) in enumerate(zip(generated_avatars, enhanced_avatars)):
        print(f"Avatar {i+1}:")
        display(gen_img)
        display(enh_img)

    # Save results
    save_images(generated_avatars, prefix="generated")
    save_images(enhanced_avatars, prefix="enhanced")

import gradio as gr

def generate_avatar(prompt):
    # Load the fine-tuned model
    pipe = StableDiffusionPipeline.from_pretrained("fine_tuned_superman", torch_dtype=torch.float16).to("cuda")

    # Generate the image
    image = pipe(prompt).images[0]
    return image

# Gradio UI
gr.Interface(
    fn=generate_avatar,
    inputs=gr.Textbox(label="Enter prompt (e.g., 'realistic Superman avatar')"),
    outputs=gr.Image(label="Generated Avatar"),
).launch()

