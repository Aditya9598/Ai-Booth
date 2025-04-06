
# Ai-Booth


This project demonstrates how to train a custom DreamBooth model using Hugging Face's Diffusers library in Google Colab. DreamBooth allows you to fine-tune Stable Diffusion to generate personalized images based on your custom images (e.g., a person, pet, object).

---

## 🚀 Features

- Train DreamBooth using your own images
- Based on Stable Diffusion v1.5
- Run entirely on Google Colab (no setup required)
- Save and reuse your trained weights

---

## 🧰 Requirements

- Python 3.8+
- Google Colab with GPU enabled
- Hugging Face account (for model access)

## 📁 Upload Your Images

Upload 3–10 images of your subject:

```python
from google.colab import files
import os

os.makedirs("train_images", exist_ok=True)
files.upload()  # Upload images to /content/train_images
```

---

## 🧠 Start DreamBooth Training

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Set training parameters
instance_prompt = "photo of sks person"
output_dir = "/content/dreambooth_weights"

# Training script goes here...
# (See notebook for full code or scripts/train.py if available)
```

---

## 💾 Save and Load Trained Weights

```python
# Save weights
pipe.unet.save_attn_procs(output_dir)

# Load later
pipe.unet.load_attn_procs(output_dir)
```

To save to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/dreambooth_weights /content/drive/MyDrive/
```

---

## 🖼️ Generate Images

```python
prompt = "photo of sks person riding a dragon"
image = pipe(prompt).images[0]
image.show()
```

---

## 📎 Resources

- [Hugging Face DreamBooth Guide](https://huggingface.co/blog/dreambooth)
- [Diffusers GitHub Repo](https://github.com/huggingface/diffusers)
- [Google Colab](https://colab.research.google.com/)

---

## 🧑‍💻 Author

**Aditya Gupta**  
[LinkedIn](https://linkedin.com/in/adityagupta-profile) | [GitHub](https://github.com/your-github-handle)

---

## 📝 License

This project is licensed under the MIT License.
