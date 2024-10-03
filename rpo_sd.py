#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import logging
import math
import os
import shutil
import json
from pathlib import Path
import sys
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from utils import (
    load_prompts, 
    remove_unique_token, 
    clean_subject, 
    training_prompts,
    validation_prompts,
)
from evaluation_metrics import DINO_score, CLIP_I_score, CLIP_T_score, RewardModel
from config.common_args import rpo_sd_args
from datasets_utils import PreferenceDataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

IS_STDOUT = not sys.stdout.isatty()

def convert(o):
    if isinstance(o, np.generic):
        return o.item()

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )
    tags = ["text-to-image", "rpo", "diffusers-training"]
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def evaluation(
    pipeline,
    args,
    accelerator,
):
    # ----------------------- Load LoRA modules ----------------------- #
    # pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

    # ----------------------- Load scheduler ----------------------- #
    # scheduler_args = {}
    # if "variance_type" in pipeline.scheduler.config:
    #     variance_type = pipeline.scheduler.config.variance_type

    #     if variance_type in ["learned", "learned_range"]:
    #         variance_type = "fixed_small"
        
    #     scheduler_args["variance_type"] = variance_type

    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    local_path = "logs/sampled_images/"
    os.makedirs(local_path, exist_ok=True)

    live_subjects = ["dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8", "cat", "cat2"]
    if args.subject not in live_subjects:
        live = False
    else:
        live = True
    test_prompts = load_prompts("[V]", args.class_token, live=live)

    logger.info(f"Running evaluation...")
    dino_score, clip_i_score, clip_t_score, rewards = [], [], [], []
    reward_model = RewardModel(args.reference_data_dir, rtype="mix")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    for test_prompt in tqdm(
        test_prompts, 
        desc="Generating images for evaluation prompts...", 
        disable=IS_STDOUT
    ):
        images = []
        # ----------------------- Inference ----------------------- #
        local_path = f"logs/sampled_images/rpo_lora/{args.subject}/"
        os.makedirs(local_path, exist_ok=True)
        pipeline_args = {"prompt": test_prompt}
        for _ in range(args.num_validation_images):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
        
        # ----------------------- Evaluation ----------------------- #
        clean_test_prompt = remove_unique_token(test_prompt, "[V]")
        dino = DINO_score(args.reference_data_dir, images)
        clip_i = CLIP_I_score(args.reference_data_dir, images)
        clip_t = CLIP_T_score([clean_test_prompt] * len(images), images)
        reward = reward_model.get_reward(images, [clean_test_prompt] * len(images))

        dino_score.append(dino)
        clip_i_score.append(clip_i)
        clip_t_score.append(clip_t)
        rewards.append(reward.mean())

        for i, image in enumerate(images):
            image_filename = local_path + f"{test_prompt}-{i}.jpg"
            image.save(image_filename)

    results = {
        "DINO Score": np.mean(dino_score),
        "CLIP I Score": np.mean(clip_i_score),
        "CLIP T Score": np.mean(clip_t_score),
        "Reward": np.mean(rewards),
    }

    file_dir = "logs/results/rpo_lora/"
    os.makedirs(file_dir, exist_ok=True)
    filename = file_dir + f"{args.subject}.json"
    with open(filename, "w") as f:
        json.dump(results, f, default=convert, indent=4)

    # blob_destination = f"lambda={args.reward_lambda}/experiments/rpo_lora/{args.subject}.json"
    # upload_file_to_gcs(filename, args.bucket, blob_destination)
    print("Evaluation is finished!")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    print("Initializing accelerator...", flush=True)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    print("Accelerator initialized.", flush=True)
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
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
    print("Generating sample images...", flush=True)

    # -------------------------- Generate sample images from the pretrained model -------------------------- #
    class_images_dir = Path(args.generated_data_dir)
    os.makedirs(class_images_dir, exist_ok=True)
    sub_folders = [name for name in os.listdir(class_images_dir)
                    if os.path.isdir(os.path.join(class_images_dir, name))]
    num_generated_folders = len(sub_folders)
    
    instance_images_dir = Path(args.reference_data_dir)
    num_prompts = len(list(instance_images_dir.iterdir()))
    subject = clean_subject(args.subject)
    live_subjects = ["dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8", "cat", "cat2"]
    live = True if subject in live_subjects else False
    generated_prompts = training_prompts("[V]", args.class_token, live=live)
    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.bfloat16

    pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
                variant=args.variant,
            )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)
    
    shutil.rmtree(class_images_dir, ignore_errors=True)
    logger.info(f"Generating training images.")
    if num_generated_folders < num_prompts:
        for prompt in tqdm(generated_prompts, desc="Generating sample images", disable=IS_STDOUT):
            print(f"Prompt = {prompt}")
            images = []
            for _ in range(args.num_validation_images):
                with torch.cuda.amp.autocast():
                    image = pipeline(prompt).images[0]
                    images.append(image)

            for _, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                os.makedirs(f"{class_images_dir}/{prompt}", exist_ok=True)
                image_filename = f"{class_images_dir}/{prompt}/{hash_image}.jpg"
                image.save(image_filename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )


    # Collate function for data loader
    def collate_fn(examples):
        input_ids = [example["prompt_ids"] for example in examples]
        input_ids += [example["desc_prompt_ids"] for example in examples]
        pixel_values = [example["pixel_values"] for example in examples]
        labels = [example["labels"] for example in examples]

        input_ids = tokenizer.pad(
            {"input_ids": input_ids}, 
            return_tensors="pt", 
            padding="max_length",
            max_length=args.tokenizer_max_length,
        ).input_ids

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        labels = torch.stack(labels)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
        }

        return batch


    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # unet.requires_grad_(False)
    ref_unet.requires_grad_(False)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

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

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.append(text_encoder)

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.pre_compute_text_embeddings:

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(args.reference_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(args.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None

        if args.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(args.class_prompt)
        else:
            pre_computed_class_prompt_encoder_hidden_states = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None
        pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation
    train_dataset = PreferenceDataset(
        reference_data_root=args.reference_data_dir,
        generated_data_root=args.generated_data_dir,
        prompt=args.reference_prompt,
        desc_prompts=generated_prompts,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        rtype="mix",
        lambda_=0.0,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

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
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("rpo", config=tracker_config)

    # Validation
    validate_prompts = validation_prompts("[V]", args.class_token)
    def validation(
        args,
        accelerator,
        unet,
    ):
        reward_list = []
        pipeline.unet = unwrap_model(unet)
        # ----------------------- Load scheduler ----------------------- #
        scheduler_args = {}
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"
            
            scheduler_args["variance_type"] = variance_type

        validation_path = f"logs/validation_images/rpo/{args.subject}/"
        os.makedirs(validation_path, exist_ok=True)
        
        reward_model = RewardModel(args.reference_data_dir, rtype="mix")
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        for prompt in validate_prompts:
            images = []
            # ----------------------- Inference ----------------------- #
            pipeline_args = {"prompt": prompt}
            for _ in range(args.num_validation_images):
                with torch.cuda.amp.autocast():
                    image = pipeline(**pipeline_args, generator=generator).images[0]
                    images.append(image)
            
            # ----------------------- Evaluation ----------------------- #
            for i, image in enumerate(images):
                image_filename = validation_path + f"{prompt}-{i}.jpg"
                image.save(image_filename)
            clean_test_prompt = remove_unique_token(prompt, "[V]")
            reward = reward_model.get_reward(images, [clean_test_prompt] * len(images), lambda_=args.reward_lambda)
            reward_list.append(reward.mean())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.mean(reward_list)

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
        disable=IS_STDOUT,
    )

    max_reward = -np.inf
    rewards_history = []
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    save_path = args.savepath
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = torch.cat(
                    torch.chunk(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype), 2, dim=1)
                )

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    None,
                )

                if unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    return_dict=False,
                )[0]

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Compute the difference for the learned model
                model_losses = torch.mean((model_pred - target) ** 2, dim=(1, 2, 3))
                model_losses_ref, model_losses_gen = torch.chunk(model_losses, 2, dim=0)
                # Get the reference prediction
                ref_model_pred = ref_unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    return_dict=False,
                )[0]
                ref_losses = torch.mean((ref_model_pred - target) ** 2, dim=(1, 2, 3))
                ref_losses_ref, ref_losses_gen = torch.chunk(ref_losses, 2, dim=0)

                # Compute the loss
                kl_diff = (ref_losses_ref - model_losses_ref) - (ref_losses_gen - model_losses_gen)

                labels = batch["labels"]
                similar_loss = torch.mean(model_losses_ref)
                preference_loss = -torch.mean(
                    labels * F.logsigmoid(kl_diff) + (1 - labels) * F.logsigmoid(-kl_diff)
                )
                loss = similar_loss + preference_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if global_step % args.validation_step == 0:
                    reward = validation(args, accelerator, unet)
                    rewards_history.append(reward)
                    logger.info(f"Validation reward: {reward:.3f}")
                    if reward > max_reward:
                        shutil.rmtree(save_path)
                        os.makedirs(save_path)

                        pipeline.save_pretrained(save_path)
                        logger.info(f"Max reward increasing from {max_reward:.3f} to {reward:.3f}, Saved state to {save_path}")
                        max_reward = reward

            if global_step >= args.max_train_steps:
                break
        

    accelerator.wait_for_everyone()
    # Save rewards history
    # reward_list = [float(x) for x in rewards_history]
    # file_dir = f"logs/results/rpo_lora/"
    # os.makedirs(file_dir, exist_ok=True)
    # reward_path = file_dir + f"{args.subject}_rewards_history.json"
    # with open(reward_path, "w") as f:
    #     json.dump(reward_list, f)
    # upload_file_to_gcs(
    #     reward_path, 
    #     args.bucket, 
    #     f"lambda={args.reward_lambda}/experiments/rpo_lora/{subject}_rewards_history.json")
    pipeline.unet = unwrap_model(unet)
    # if accelerator.is_main_process:
    #     evaluation(pipeline, args, accelerator)
    shutil.rmtree(class_images_dir)
    shutil.rmtree("logs/validation_images")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPO of a training script.")
    args = rpo_sd_args(parser)
    main(args)