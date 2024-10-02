from PIL import Image
from pathlib import Path
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_images
from evaluation_metrics import RewardModel
import jax
import os

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
        tokenizer,
        size = 512,
        center_crop=False,
        rtype="i2i",
        lambda_=0.3,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # Load the reference images
        self.reference_data_root = Path(reference_data_root)
        if not self.reference_data_root.exists():
            raise ValueError("reference images root doesn't exists.")

        self.reference_images_path = list(Path(reference_data_root).iterdir())
        self.num_reference_images = len(self.reference_images_path)
        self.prompt = prompt
        self.desc_prompts = desc_prompts
        self._length = self.num_reference_images

        # Load the generated images
        self.generated_data_root = generated_data_root
        if not Path(generated_data_root).exists():
            raise ValueError("Generated images root doesn't exists.")
        
        self.generated_data_root = generated_data_root
        self.num_generated_images = jax.device_count()
        self._length = max(self.num_generated_images , self.num_reference_images)

        # Compute the reward
        # reference_images = read_images(reference_data_root)
        # generated_images = read_images(generated_data_root)
        self.reward_model = RewardModel(reference_data_root, rtype=rtype)
        self.reference_image_rewards = {}
        self.generated_image_rewards = {}
        reference_images = read_images(reference_data_root)
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
        # self.reference_rewards = reward_model.get_reward(reference_images)
        # self.generated_rewards = reward_model.get_reward(generated_images)
        # self.rewards = reward_fn(reference_images, generated_images)

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
        example["prompt_ids"] = self.tokenizer(
            self.prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        reference_image = Image.open(self.reference_images_path[index % self.num_reference_images])
        if not reference_image.mode == "RGB":
            reference_image = reference_image.convert("RGB")

        desc_prompt = self.desc_prompts[index % len(self.desc_prompts)]
        example["desc_prompt_ids"] = self.tokenizer(
            desc_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

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

        # labels = torch.ones(1)
        # reference_rewards = self.reference_rewards[index % self.num_reference_images]
        # generated_rewards = self.generated_rewards[index % self.num_generated_images]
        reference_rewards = self.reference_image_rewards[desc_prompt]
        generated_rewards = self.generated_image_rewards[desc_prompt][index % self.num_generated_images]
        difference = (reference_rewards - generated_rewards)
        difference = torch.from_numpy(np.array(difference))
        probs = torch.sigmoid(difference)
        labels = torch.bernoulli(probs)
        example["labels"] = labels

        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example
    
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the reference and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        reference_data_root,
        reference_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.reference_data_root = Path(reference_data_root)
        if not self.reference_data_root.exists():
            raise ValueError("reference images root doesn't exists.")

        self.reference_images_path = list(Path(reference_data_root).iterdir())
        self.num_reference_images = len(self.reference_images_path)
        self.reference_prompt = reference_prompt
        self._length = self.num_reference_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_reference_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

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
        reference_image = Image.open(self.reference_images_path[index % self.num_reference_images])
        if not reference_image.mode == "RGB":
            reference_image = reference_image.convert("RGB")
        example["reference_images"] = self.image_transforms(reference_image)
        example["reference_prompt_ids"] = self.tokenizer(
            self.reference_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example