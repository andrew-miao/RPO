import argparse
import torch
from huggingface_hub.utils import insecure_hashlib
import os
from accelerate import PartialState
from diffusers import StableDiffusionPipeline

def inference(pipeline, prompt, save_dir, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    generator = torch.torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    for _ in range(16):
        with torch.no_grad():
            image = pipeline(prompt, generator=generator).images[0]
            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = os.path.join(save_dir, f"{prompt}-{hash_image}.png")
            image.save(image_filename)

def parallel_inference(pipeline, prompt, save_dir):
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)
    ngpus = torch.cuda.device_count()

    with distributed_state.split_between_processes([prompt] * ngpus) as prompts:
        image = pipeline(prompts).images[0]
        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
        image_filename = os.path.join(save_dir, f"{prompt}-{hash_image}.png")
        image.save(image_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a prompt using a Stable Diffusion model.")
    parser.add_argument("--pretrained_model_path", type=str, default="logs/path-to-save-model/rpo", help="The path of the pretrained model.")
    parser.add_argument("--prompt", type=str, default="a photo of a green colered apple", help="The prompt to generate the image.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible inference.")
    parser.add_argument("--save_dir", type=str, default="logs/inference/dog", help="The directory to save the generated images.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    args = parser.parse_args()
    return args

def main(args):
    prompts = args.prompt
    save_dir = os.path.join(args.save_dir, args.prompt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_path, torch_dtype=weight_dtype)

    print("Pipeline loaded", flush=True)
    print("Starting inference...", flush=True)
    if torch.cuda.device_count() > 1:
        parallel_inference(pipeline, prompts, save_dir)
    else:
        inference(pipeline, prompts, save_dir, args)

    print(f"Images saved to {save_dir}", flush=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)