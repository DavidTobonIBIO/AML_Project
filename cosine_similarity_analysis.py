import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DPMSolverSinglestepScheduler


parser = argparse.ArgumentParser()
parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
parser.add_argument('--seed', type=int, default=32, help='Random seed')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
parser.add_argument('--prompt', type=str, default="Lovely white fondant-covered cake with a purple ribbon", help='Text prompt')
parser.add_argument('--output_folder', type=str, default="cosine_similarity_results", help='Output folder')

args = parser.parse_args()

# Create output folder
os.makedirs(args.output_folder, exist_ok=True)
os.makedirs(os.path.join(args.output_folder, "images"), exist_ok=True)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# Load models
print("Loading models...")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = DPMSolverSinglestepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# Generation parameters
height = 512
width = 512
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
batch_size = 1

print(f"\nGenerating with prompt: '{args.prompt}'")
print(f"Guidance scale: {guidance_scale}")
print(f"Inference steps: {num_inference_steps}")

# Prepare text embeddings
generator = torch.manual_seed(args.seed)
text_input = tokenizer(args.prompt, padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings_combined = torch.cat([uncond_embeddings, text_embeddings])

# Initialize latents
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

# Storage for cosine similarity values
cosine_similarities = []
timesteps_list = []

# Function to compute cosine similarity
def compute_cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors (flattened).
    """
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    return similarity.item()

print("\nStarting generation and computing cosine similarities...")

# Denoising loop
for k, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
    # Expand latents for CFG
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # Predict noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_combined).sample
    
    # Split predictions
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # Compute cosine similarity
    cos_sim = compute_cosine_similarity(noise_pred_uncond, noise_pred_text)
    cosine_similarities.append(cos_sim)
    timesteps_list.append(t.item())
    
    # Apply CFG
    noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Compute previous sample
    latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

print("\nCosine Similarity Results:")
print("=" * 60)
print(f"{'Step':<8} {'Timestep':<12} {'Cosine Similarity':<20}")
print("=" * 60)
for i, (ts, sim) in enumerate(zip(timesteps_list, cosine_similarities)):
    print(f"{i+1:<8} {ts:<12.2f} {sim:<20.6f}")
print("=" * 60)
print(f"Mean: {np.mean(cosine_similarities):.6f}")
print(f"Std:  {np.std(cosine_similarities):.6f}")
print(f"Min:  {np.min(cosine_similarities):.6f}")
print(f"Max:  {np.max(cosine_similarities):.6f}")

# Save statistics to file
stats_file = os.path.join(args.output_folder, "cosine_similarity_stats.txt")
with open(stats_file, 'w') as f:
    f.write(f"Prompt: {args.prompt}\n")
    f.write(f"Guidance Scale: {guidance_scale}\n")
    f.write(f"Inference Steps: {num_inference_steps}\n")
    f.write(f"Seed: {args.seed}\n\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Step':<8} {'Timestep':<12} {'Cosine Similarity':<20}\n")
    f.write("=" * 60 + "\n")
    for i, (ts, sim) in enumerate(zip(timesteps_list, cosine_similarities)):
        f.write(f"{i+1:<8} {ts:<12.2f} {sim:<20.6f}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Mean: {np.mean(cosine_similarities):.6f}\n")
    f.write(f"Std:  {np.std(cosine_similarities):.6f}\n")
    f.write(f"Min:  {np.min(cosine_similarities):.6f}\n")
    f.write(f"Max:  {np.max(cosine_similarities):.6f}\n")

print(f"\nStatistics saved to: {stats_file}")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Cosine Similarity vs Step
axes[0].plot(range(1, len(cosine_similarities) + 1), cosine_similarities, 
             marker='o', linewidth=2, markersize=6, color='#2E86AB')
axes[0].axhline(y=np.mean(cosine_similarities), color='red', linestyle='--', 
                linewidth=1.5, label=f'Mean: {np.mean(cosine_similarities):.4f}')
axes[0].set_xlabel('Denoising Step', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
axes[0].set_title('Cosine Similarity between Conditional and Unconditional Predictions\n(vs Denoising Step)', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_ylim([min(cosine_similarities) - 0.05, max(cosine_similarities) + 0.05])

# Plot 2: Cosine Similarity vs Timestep
axes[1].plot(timesteps_list, cosine_similarities, 
             marker='s', linewidth=2, markersize=6, color='#A23B72')
axes[1].axhline(y=np.mean(cosine_similarities), color='red', linestyle='--', 
                linewidth=1.5, label=f'Mean: {np.mean(cosine_similarities):.4f}')
axes[1].set_xlabel('Timestep (Noise Level)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
axes[1].set_title('Cosine Similarity between Conditional and Unconditional Predictions\n(vs Timestep)', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].invert_xaxis()  # Higher timesteps (more noise) on the left
axes[1].set_ylim([min(cosine_similarities) - 0.05, max(cosine_similarities) + 0.05])

plt.tight_layout()
plot_file = os.path.join(args.output_folder, "cosine_similarity_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_file}")
plt.close()

# Create additional detailed plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

steps = range(1, len(cosine_similarities) + 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(cosine_similarities)))

bars = ax.bar(steps, cosine_similarities, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=np.mean(cosine_similarities), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(cosine_similarities):.4f}')

ax.set_xlabel('Denoising Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
ax.set_title(f'Cosine Similarity Evolution During CFG\nPrompt: "{args.prompt[:50]}..."', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([min(cosine_similarities) - 0.05, 1.0])

plt.tight_layout()
bar_plot_file = os.path.join(args.output_folder, "cosine_similarity_bars.png")
plt.savefig(bar_plot_file, dpi=300, bbox_inches='tight')
print(f"Bar plot saved to: {bar_plot_file}")
plt.close()

# Generate and save the final image
print("\nGenerating final image...")
latents = 1 / 0.18215 * latents

with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8").squeeze()

image = Image.fromarray(image)
image_file = os.path.join(args.output_folder, "images", f"generated_image_seed_{args.seed}.png")
image.save(image_file)
print(f"Generated image saved to: {image_file}")

print(f"\n✓ Analysis complete! All results saved to '{args.output_folder}/'")
