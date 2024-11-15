import torch
import os
from diffusers import StableDiffusionXLPipeline
import argparse
from accelerate import PartialState
from huggingface_hub.utils import insecure_hashlib


def load_pipeline(lora_dir):
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")    
    pipeline.load_lora_weights(lora_dir)  
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

def inference(pipeline, prompt, save_dir, seed, num_inference_images=4):
    ngpus = torch.cuda.device_count()
    generator = torch.torch.Generator(device="cuda")
    generator.manual_seed(seed)

    if ngpus > 1:
        distributed_state = PartialState()
        pipeline.to(distributed_state.device)
        with distributed_state.split_between_processes([prompt] * ngpus) as prompt:
            image = pipeline(prompt=prompt, generator=generator).images[0]
            hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
            image_filename = os.path.join(save_dir, f"{prompt}-{hash_image}.png")
            image.save(image_filename)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        for _ in range(num_inference_images):
            with torch.no_grad():
                image = pipeline(prompt=prompt, generator=generator).images[0]
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = os.path.join(save_dir, f"{prompt}-{hash_image}.png")
                image.save(image_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="logs/rpo_sdxl/output/checkpoint-1000",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="dog",
        help="The subject of the prompt."
    )
    parser.add_argument(
        "--num_inference_images",
        type=int,
        default=16,
        help="The number of images to generate for each prompt."
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="logs/inference/rpo_sdxl/inference/",
        help="The directory to save the generated images",
    )

    args = parser.parse_args()
    return args

def main(args):
    prompt = args.prompt

    pipeline = load_pipeline(args.pretrained_model_name_or_path)
    save_dir = args.save_dir + f"{args.subject}/"
    os.makedirs(save_dir, exist_ok=True)
    inference(pipeline, prompt, save_dir, args.seed, args.num_inference_images)
    print(f"Inference images saved to {save_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)