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
parser.add_argument('--conditional_threshold', type=float, default=0.9, help='Threshold for switching to conditional only (0.9 = 90% of process)')

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
text_input = tokenizer(args.prompt, padding="max_length", max_length=tokenizer.model_max_length, 
                       truncation=True, return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings_combined = torch.cat([uncond_embeddings, text_embeddings])

# Function to compute cosine similarity
def compute_cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors (flattened).
    """
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
    return similarity.item()

# Function to generate image
def generate_image(seed, use_cfg=True, conditional_threshold=1.0):
    """
    Generate an image with either full CFG or late-stage conditional only.
    
    Args:
        seed: Random seed
        use_cfg: If True, use CFG throughout. If False, use conditional only after threshold.
        conditional_threshold: Fraction of process after which to use conditional only (0.9 = 90%)
    """
    generator = torch.manual_seed(seed)
    
    # Initialize latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    
    cosine_similarities = []
    timesteps_list = []
    
    # Calculate the step threshold for switching
    threshold_step = int(num_inference_steps * conditional_threshold)
    
    for k, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps), 
                     desc=f"Generating {'CFG' if use_cfg else f'Late-stage conditional (>{conditional_threshold*100:.0f}%)'}" ):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_combined).sample
        
        # Split predictions
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Compute cosine similarity (only for the first generation)
        if use_cfg:
            cos_sim = compute_cosine_similarity(noise_pred_uncond, noise_pred_text)
            cosine_similarities.append(cos_sim)
            timesteps_list.append(t.item())
        
        # Decide which noise prediction to use
        if use_cfg:
            # Full CFG throughout
            noise_pred_final = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            # Use conditional only after threshold, CFG before
            if k >= threshold_step:
                # After threshold: use text-conditional only
                noise_pred_final = noise_pred_text
            else:
                # Before threshold: use CFG
                noise_pred_final = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous sample
        latents = scheduler.step(noise_pred_final, t, latents).prev_sample
    
    # Decode latents to image
    latents = 1 / 0.18215 * latents
    
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8").squeeze()
    
    return Image.fromarray(image), cosine_similarities, timesteps_list

# Generate CFG image and compute cosine similarities
print("\n" + "="*70)
print("PHASE 1: Generating with full CFG and computing cosine similarities")
print("="*70)
image_cfg, cosine_similarities, timesteps_list = generate_image(args.seed, use_cfg=True)

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

# Create cosine similarity line plot
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

steps = range(1, len(cosine_similarities) + 1)
ax.plot(steps, cosine_similarities, marker='o', linewidth=2.5, markersize=7, 
        color='#2E86AB', label='Cosine Similarity')
ax.axhline(y=np.mean(cosine_similarities), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(cosine_similarities):.4f}')

ax.set_xlabel('Denoising Step', fontsize=13, fontweight='bold')
ax.set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
ax.set_title('Cosine Similarity between Conditional and Unconditional UNet Predictions', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.legend(fontsize=11, loc='best')
ax.set_ylim([0, 1])  # Set y-axis limits from 0 to 1
ax.set_xlim([0, len(cosine_similarities) + 1])

plt.tight_layout()
plot_file = os.path.join(args.output_folder, "cosine_similarity_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Cosine similarity plot saved to: {plot_file}")
plt.close()

# Save CFG image
image_cfg_file = os.path.join(args.output_folder, "images", f"cfg_full_seed_{args.seed}.png")
image_cfg.save(image_cfg_file)
print(f"CFG image saved to: {image_cfg_file}")

# Generate late-stage conditional image
print("\n" + "="*70)
print(f"PHASE 2: Generating with late-stage conditional (>{args.conditional_threshold*100:.0f}%)")
print("="*70)
image_late_cond, _, _ = generate_image(args.seed, use_cfg=False, conditional_threshold=args.conditional_threshold)

# Save late-stage conditional image
image_late_cond_file = os.path.join(args.output_folder, "images", f"late_stage_conditional_seed_{args.seed}.png")
image_late_cond.save(image_late_cond_file)
print(f"Late-stage conditional image saved to: {image_late_cond_file}")

# Create comparison grid
print("\nCreating comparison grid...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Display CFG image
axes[0].imshow(image_cfg)
axes[0].set_title(f'Full CFG (guidance_scale={guidance_scale})\nSeed: {args.seed}', 
                  fontsize=12, fontweight='bold', pad=10)
axes[0].axis('off')

# Display late-stage conditional image
axes[1].imshow(image_late_cond)
threshold_step = int(num_inference_steps * args.conditional_threshold)
axes[1].set_title(f'Late-Stage Conditional Only\n(Conditional after step {threshold_step}/{num_inference_steps})\nSeed: {args.seed}', 
                  fontsize=12, fontweight='bold', pad=10)
axes[1].axis('off')

plt.suptitle(f'Comparison: Full CFG vs Late-Stage Conditional\nPrompt: "{args.prompt}"', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

comparison_file = os.path.join(args.output_folder, "comparison_grid.png")
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"Comparison grid saved to: {comparison_file}")
plt.close()

print(f"\n" + "="*70)
print("✓ Analysis complete! All results saved to '{}'" .format(args.output_folder))
print("="*70)
print(f"\nGenerated files:")
print(f"  - {stats_file}")
print(f"  - {plot_file}")
print(f"  - {image_cfg_file}")
print(f"  - {image_late_cond_file}")
print(f"  - {comparison_file}")
