
import argparse
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import torch
import torch.nn.functional as F

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers

from evaluation_metrics import RewardModel
from utils import read_images, training_prompts, clean_subject
from inference_sdxl import inference

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=args.cache_dir
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # pretrained model config
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store the pre-trained models downloaded from Hugging Face.")
    # data config
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--config_name", type=str, default="")
    # validation config
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is used during validation to verify that the model is learning.",)
    parser.add_argument("--num_validation_images", type=int, default=0, help="Number of images that should be generated during validation with `validation_prompt`.",)
    parser.add_argument("--validation_epochs", type=int, default=50000)
    # use prior preservation
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.",)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    # save config
    parser.add_argument("--output_dir", type=str, default="outdir", help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--class_token", type=str, default="dog", help="The subject of the images.")
    parser.add_argument("--prompt", type=str, default="A photo of a [V] dog", help="The prompt to use for training.")
    # dataloader config
    parser.add_argument("--reference_data_dir", type=str, default="../dreambooth/dataset/dog", help="The directory where the reference images are stored.",)
    parser.add_argument("--generated_data_dir", type=str, default="/scratch/ssd004/scratch/yanting", required=True, help="The directory where the generated images are stored.",)
    parser.add_argument("--subfolder", type=str, default="dog", help="The subfolder of the reference images.",)
    parser.add_argument("--resolution", type=int, default=1024, help="The resolution for input images, all the images in the train/validation dataset will be resized to this")
    parser.add_argument("--crops_coords_top_left_h", type=int, default=0, help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),)
    parser.add_argument("--crops_coords_top_left_w", type=int, default=0, help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),)
    parser.add_argument("--center_crop", default=False, action="store_true", help=("Whether to center crop the input images to the resolution. If not set, the images will be randomly"    " cropped. The images will be resized to the resolution first before cropping."),)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."),)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint.")
    # train config
    parser.add_argument("--beta", type=float, default=1.0, help="The beta value for the KL loss.",)
    parser.add_argument("--train_text_encoder_ti", action="store_true", help=("Whether to use textual inversion"),)
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",)    
    # optimizer config
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6, help="Text encoder learning rate to use.",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'    ' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    # optimizer config
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # save config
    parser.add_argument("--logging_dir", type=str, default="logs", help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"    " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),)
    parser.add_argument("--allow_tf32", action="store_true", help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"    " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
    parser.add_argument("--report_to", type=str, default="tensorboard", help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'    ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="    " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"    " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--rank", type=int, default=32, help=("The dimension of the LoRA update matrices."),)
    parser.add_argument("--offset_noise", type=float, default=0.0)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# Taken from https://github.com/replicate/cog-sdxl/blob/main/dataset_and_utils.py
class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids = None
        self.inserting_tokens = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_tokens, initializer_tokens):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_tokens, list), "inserting_tokens should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_tokens
            ), "All elements in inserting_tokens should be strings."

            self.inserting_tokens = inserting_tokens
            special_tokens_dict = {"additional_special_tokens": self.inserting_tokens}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_tokens)
            std_token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data.std()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding
            print(f"{idx} text encodedr's std_token_embedding: {std_token_embedding}")

            embeddings = []
            embeddings_norm = []
            for initializer_token in initializer_tokens:
                if initializer_token == "":
                    emb = torch.randn(1, text_encoder.text_model.config.hidden_size).to(device=self.device).to(dtype=self.dtype) * std_token_embedding
                    embeddings.append(emb)
                    embeddings_norm.append(std_token_embedding)
                else:
                    initializer_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)
                    emb = text_encoder.text_model.embeddings.token_embedding.weight.data[initializer_token_id]
                    embeddings.append(emb)
                    embeddings_norm.append(emb.norm().item())
            
            embeddings = torch.cat(embeddings, dim=0)
            text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = embeddings
            embeddings_norm = torch.tensor(embeddings_norm).unsqueeze(1)
            self.embeddings_settings[f"token_embedding_norm_{idx}"] = embeddings_norm
    
            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu
            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_0 - CLIP ViT-L/14, text_encoder_1 -  CLIP ViT-G/14
        idx_to_text_encoder_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[0] == len(
                self.tokenizers[0]
            ), "Tokenizers should be the same."
            new_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids]
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            index_updates = ~index_no_updates
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]
            new_embeddings = F.normalize(new_embeddings, dim=-1) * self.embeddings_settings[f"token_embedding_norm_{idx}"].view(-1, 1).to(device=text_encoder.device)
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates] = new_embeddings.to(device=text_encoder.device).to(dtype=text_encoder.dtype)

class PreferenceDataset(Dataset):
    """
    A dataset to prepare the reference and generated images with the prompts for fine-tuning the model.
    """
    def __init__(
        self,
        reference_data_root,
        generated_data_root,
        prompt,
        desc_prompts,
        size=1024,
        center_crop=False,
        rtype="mix",
        lambda_=0.0,
    ):
        self.size = size
        self.center_crop = center_crop

        # Load the reference images
        self.reference_data_root = Path(reference_data_root)
        if not self.reference_data_root.exists():
            raise ValueError("reference images root doesn't exists.")

        reference_images_path = list(Path(reference_data_root).iterdir())
        self.reference_images_path = []
        image_extensions = [".jpg", ".jpeg", ".png"]
        for file in reference_images_path:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in image_extensions:
                self.reference_images_path.append(file)

        reference_images = read_images(reference_data_root)
        self.num_reference_images = len(reference_images)

        self.prompt = prompt
        self.desc_prompts = desc_prompts
        self._length = self.num_reference_images

        # Load the generated images
        self.generated_data_root = generated_data_root
        if not Path(generated_data_root).exists():
            raise ValueError("Generated images root doesn't exists.")
        
        self.generated_data_root = generated_data_root
        self.num_generated_images = 4
        self._length = max(self.num_generated_images, self.num_reference_images)

        # Compute the reward
        self.reward_model = RewardModel(reference_data_root, rtype=rtype)
        self.reference_image_rewards = {}
        self.generated_image_rewards = {}

        for i in range(len(desc_prompts)):
            desc_prompt = desc_prompts[i]
            generated_images_path = os.path.join(self.generated_data_root, desc_prompt)
            generated_images = read_images(generated_images_path)
            rewards = self.reward_model.get_reward(
                [reference_images[i % self.num_reference_images]] + generated_images, [desc_prompt] * (1 + len(generated_images)),
                lambda_=lambda_,
            )

            self.reference_image_rewards[desc_prompt] = rewards[0]
            self.generated_image_rewards[desc_prompt] = rewards[1:]

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        example["prompts"] = self.prompt

        reference_image = Image.open(self.reference_images_path[index % self.num_reference_images])
        if not reference_image.mode == "RGB":
            reference_image = reference_image.convert("RGB")

        desc_prompt = self.desc_prompts[index % len(self.desc_prompts)]
        example["desc_prompts"] = desc_prompt

        generated_images_path = list(Path(os.path.join(self.generated_data_root, desc_prompt)).iterdir())
        n_generated_images = len(generated_images_path)
        generated_image = Image.open(generated_images_path[index % n_generated_images])

        if not generated_image.mode == "RGB":
            generated_image = generated_image.convert("RGB")

        pixel_values = torch.cat(
            (self.image_transforms(reference_image), 
            self.image_transforms(generated_image)),
            dim=0
        )
        example["pixel_values"] = pixel_values
        reference_rewards = self.reference_image_rewards[desc_prompt]
        generated_rewards = self.generated_image_rewards[desc_prompt][index % self.num_generated_images]
        difference = (reference_rewards - generated_rewards)
        if isinstance(difference, np.ndarray):
            difference = torch.from_numpy(difference)
        if not isinstance(difference, torch.Tensor):
            difference = torch.tensor(difference)

        probs = torch.sigmoid(difference)
        labels = torch.bernoulli(probs)
        example["labels"] = labels

        return example


def collate_fn(examples):
    prompts = [example["prompts"] for example in examples]
    prompts += [example["desc_prompts"] for example in examples]
    pixel_values = [example["pixel_values"] for example in examples]
    labels = [example["labels"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    labels = torch.stack(labels)

    batch = {
        "prompts": prompts,
        "pixel_values": pixel_values,
        "labels": labels,
    }
    return batch

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

     # Generate class images if prior preservation is enabled.
    generate_images_dir = args.generated_data_dir
    logger.info(f"Generated images directory: {generate_images_dir}")
    subject = clean_subject(args.subfolder)
    live = True if subject in ["dog", "cat"] else False
    logger.info(f"Current subject: {subject}")
    generated_prompts = training_prompts("[V]", subject, live)

    if not os.path.exists(generate_images_dir):
        os.makedirs(generate_images_dir)
    # cur_class_images = len(list(generate_images_dir.iterdir()))

    # if cur_class_images < args.num_class_images:
    has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
    torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
    if args.mixed_precision == "fp32":
        torch_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )
    pipeline.set_progress_bar_config(disable=True)

    # num_new_images = args.num_class_images - cur_class_images
    # logger.info(f"Number of class images to sample: {num_new_images}.")

    # ------------------------ Generate sample images from pretrained model ------------------------ #
    if not os.path.exists(generate_images_dir):
        os.makedirs(generate_images_dir)

    sub_folders = [name for name in os.listdir(Path(generate_images_dir)) 
                   if os.path.isdir(os.path.join(Path(generate_images_dir), name))]
    num_genreated_folders = len(sub_folders)

    reference_images_dir = args.reference_data_dir
    num_prompts = len(list(Path(reference_images_dir).iterdir()))
    subject = clean_subject(args.subfolder)
    live = True if subject in ["dog", "cat"] else False
    logger.info(f"Current subject: {subject}")
    generated_prompts = training_prompts("[V]", subject, live)
    logger.info(f"training prompts = {generated_prompts}")
    if num_genreated_folders < num_prompts:
        pipeline.set_progress_bar_config(disable=True)

        num_samples = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_new_images = (num_prompts - num_genreated_folders) * num_samples
        logger.info(f"Number of generated images to sample: {num_new_images}.")
        
        for i in tqdm(
            range(len(generated_prompts)), 
            desc="Generating images", 
            disable=not accelerator.is_local_main_process
        ):  
            text = generated_prompts[i]
            gen_image_save_dir = os.path.join(generate_images_dir, text)
            os.makedirs(gen_image_save_dir, exist_ok=True)
            save_dir = gen_image_save_dir + f"/{args.subfolder}"
            inference(pipeline, text, save_dir, args.seed)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Finished generating images!")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
        cache_dir=args.cache_dir,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
        cache_dir=args.cache_dir,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir,)    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )
    vae_scaling_factor = vae.config.scaling_factor
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    ref_unet.requires_grad_(False)

    if args.train_text_encoder_ti:
        with open(args.config_dir, 'r') as data_config:
            data_cfg = json.load(data_config)[args.config_name]
            inserting_tokens = data_cfg["inserting_tokens"]
            initializer_tokens = data_cfg["initializer_tokens"]
        
        logger.info(f"List of token identifiers: {inserting_tokens}")
        # initialize the new tokens for textual inversion
        embedding_handler = TokenEmbeddingsHandler(
            [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
        )
        embedding_handler.initialize_new_tokens(
            inserting_tokens=inserting_tokens, 
            initializer_tokens=initializer_tokens
        )
            
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)
    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
    # if we use textual inversion, we freeze all parameters except for the token embeddings
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []
        for name, param in text_encoder_one.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                param = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        text_lora_parameters_two = []
        for name, param in text_encoder_two.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                param = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_two.append(param)
            else:
                param.requires_grad = False

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet, ref_unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        for model in models:
            for param in model.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    if args.train_text_encoder:
                        text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    if args.train_text_encoder:
                        text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

        text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
        )

        text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # Optimization parameters    
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}

    if not freeze_text_encoder:
        # different learning rate for text encoder and unet
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    # Optimizer creation
    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = PreferenceDataset(
        reference_data_root=args.reference_data_dir,
        generated_data_root=generate_images_dir,
        prompt=args.prompt,
        desc_prompts=generated_prompts,
        size=args.resolution,
        rtype="mix",
        lambda_=0.0,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    # Handle instance prompt.
    instance_time_ids = compute_time_ids()
    add_time_ids = instance_time_ids
    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        add_time_ids = torch.cat([add_time_ids, class_time_ids], dim=0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if not freeze_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("fine-tune sdxl", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            text_encoder_one.train()
            text_encoder_two.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            if args.train_text_encoder:
                text_encoder_one.text_model.embeddings.requires_grad_(True)
                text_encoder_two.text_model.embeddings.requires_grad_(True)

        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                # encode batch prompts when custom prompts are provided for each image -
                # if train_dataset.custom_instance_prompts:
                if freeze_text_encoder:
                    prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts)

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                pixel_values_ref, pixel_values_gen = torch.chunk(pixel_values, 2, dim=1)
                pixel_values = torch.cat([pixel_values_ref, pixel_values_gen], dim=0)
                model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = model_input * vae_scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                noise = noise + args.offset_noise * torch.randn(model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                timesteps = timesteps.chunk(2)[0].repeat(2)
                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the model input according to the noise magnitude at each timestep
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                elems_to_repeat_text_embeds = 1
                elems_to_repeat_time_ids = bsz // 2

                # Predict the noise residual
                if freeze_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.repeat(2, 1),
                        "text_embeds": unet_add_text_embeds,
                    }
                    # prompt_embeds_input = prompt_embeds
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        # prompt_embeds_input,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                    with torch.no_grad():
                        unet_added_conditions = {
                            "time_ids": add_time_ids.repeat(2, 1),
                            "text_embeds": unet_add_text_embeds.to(ref_unet.dtype),
                        }
                        model_ref_pred = ref_unet(
                            noisy_model_input.to(ref_unet.dtype), 
                            timesteps, 
                            # prompt_embeds_input,
                            prompt_embeds.to(ref_unet.dtype),
                            added_cond_kwargs=unet_added_conditions,
                        ).sample
                else:
                    unet_added_conditions = {"time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1)}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds_input, added_cond_kwargs=unet_added_conditions
                    ).sample
                    if args.beta > 0.0:
                        base_prompts = batch["base_prompts"]
                        with torch.no_grad():
                            base_prompt_embeds, base_add_embeds = compute_text_embeddings(
                                base_prompts, text_encoders, tokenizers
                            )
                            cross_attention_kwargs = {"scale": 0.0}
                            base_added_conditions = {"time_ids": add_time_ids, "text_embeds": base_add_embeds}
                            refer_pred = unet(
                                noisy_model_input, 
                                timesteps, 
                                base_prompt_embeds, 
                                added_cond_kwargs=base_added_conditions, 
                                cross_attention_kwargs=cross_attention_kwargs
                            ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute loss
                model_losses = torch.mean((model_pred - target) ** 2, dim=(1, 2, 3))
                model_losses_ref, model_losses_gen = torch.chunk(model_losses, 2, dim=0)

                model_ref_losses = torch.mean((model_ref_pred - target) ** 2, dim=(1, 2, 3))
                model_ref_losses_ref, model_ref_losses_gen = torch.chunk(model_ref_losses, 2, dim=0)

                kl_diff = (model_ref_losses_ref - model_losses_ref) - (model_ref_losses_gen - model_losses_gen)
                
                labels = batch["labels"]
                similar_loss = torch.mean(model_losses_ref)
                preference_loss = -torch.mean(
                    labels * F.logsigmoid(args.beta * kl_diff) + (1 - labels) * F.logsigmoid(-args.beta * kl_diff)
                )
                loss = similar_loss + preference_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if (args.train_text_encoder or args.train_text_encoder_ti)
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                if freeze_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        revision=args.revision,
                        variant=args.variant,
                        cache_dir=args.cache_dir,
                    )
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder_2",
                        revision=args.revision,
                        variant=args.variant,
                        cache_dir=args.cache_dir,
                    )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    cache_dir=args.cache_dir,
                )

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                pipeline_args = {"prompt": args.validation_prompt}

                with torch.cuda.amp.autocast():
                    images = [
                        pipeline(**pipeline_args, generator=generator).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            # Final inference
            # Load previous pipeline
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
            )

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}

            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type

                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"

                scheduler_args["variance_type"] = variance_type

            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

            # run inference
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")

        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(
                f"{args.output_dir}/learned_embeds.safetensors",
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)