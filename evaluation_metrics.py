import numpy as np
import transformers
import torch
import time
from functools import partial
from torchmetrics.functional.multimodal import clip_score
from utils import read_images, cosine_similarity

def DINO_score(reference_image_dir, compare_images):
    # Load the images
    reference_images = read_images(reference_image_dir)
    
    # Load the model
    processor = transformers.AutoImageProcessor.from_pretrained("facebook/dino-vits16")
    model = transformers.AutoModel.from_pretrained("facebook/dino-vits16")

    # Evaluation
    similarity = np.zeros((len(reference_images), len(compare_images)))
    for i, reference in enumerate(reference_images):
        images = [reference] + compare_images
        inputs = processor(images=images, return_tensors="pt")
        features = model(**inputs).last_hidden_state[:, 0, :]
        reference_features = features[0]
        compare_features = features[1:]
        reference_features = reference_features.cpu().detach().numpy()
        compare_features = compare_features.cpu().detach().numpy()
        reference_features = reference_features[None, :]
        similarity[i] = cosine_similarity(reference_features, compare_features) 
    
    return np.mean(similarity)

def CLIP_I_score(reference_image_dir, compare_images):
    # Load the images
    reference_images = read_images(reference_image_dir)

    # Load the model
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = transformers.FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch16")

    # Evaluation
    similarity = np.zeros((len(reference_images), len(compare_images)))
    for i, reference in enumerate(reference_images):
        images = [reference] + compare_images
        inputs = processor(images=images, return_tensors="np")
        features = model.get_image_features(**inputs)
        reference_features = features[0]
        compare_features = features[1:]
        similarity[i] = cosine_similarity(reference_features, compare_features) 
    
    return np.mean(similarity)

def CLIP_T_score(prompts, compare_images):
    # Load the images
    np_compare_images = np.asarray(compare_images)
    np_compare_images = np_compare_images.astype("uint8")
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    similarity = clip_score_fn(torch.from_numpy(np_compare_images), prompts).detach() / 100
    similarity = similarity.cpu().detach().numpy()
    return similarity.mean()


class RewardModel:
    def __init__(self, reference_dir, rtype="i2i") -> None:
        self.reference_images = read_images(reference_dir)
        self.model = transformers.AlignModel.from_pretrained("kakaobrain/align-base")
        self.processor = transformers.AutoProcessor.from_pretrained("kakaobrain/align-base")
        self.rtype = rtype

    def i2i_similarity_fn(self, compare_images):
        all_similarity = torch.zeros((len(self.reference_images), len(compare_images)))
        for i, reference_image in enumerate(self.reference_images):
            images = [reference_image] + compare_images
            inputs = self.processor(images=images, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                reference_features = features[0]
                compare_features = features[1:]

                similarity = torch.nn.functional.cosine_similarity(
                    reference_features, compare_features, dim=-1)
                all_similarity[i, :] = similarity
        all_similarity = all_similarity.cpu().detach().numpy()
        mean_similarity = np.mean(all_similarity, axis=0)
        return mean_similarity
    
    def t2i_similarity_fn(self, compare_images, prompts):
        with torch.no_grad():
            prompt_embs = self.model.get_text_features(**self.processor(
                text=prompts, return_tensors="pt")
            ).squeeze()
            input_images = self.processor(images=compare_images, return_tensors="pt")
            image_features = self.model.get_image_features(**input_images)

            similarity = torch.nn.functional.cosine_similarity(
                prompt_embs, image_features, dim=-1
            )
 
        return similarity.cpu().detach().numpy()
    
    def mix_up_similarity_fn(self, compare_images, prompts, lambda_):
        assert lambda_ >= 0 and lambda_ <= 1
        i2i_similarity = self.i2i_similarity_fn(compare_images)
        t2i_similarity = self.t2i_similarity_fn(compare_images, prompts)
        i2i_similarity = (i2i_similarity + 1) / 2
        t2i_similarity = (t2i_similarity + 1) / 2
        if lambda_ == 1:
            return i2i_similarity
        if lambda_ == 0:
            return t2i_similarity
        mix_reward = 1 / (lambda_ / (i2i_similarity + 1e-08) 
                          + (1 - lambda_) / (t2i_similarity + 1e-08))
        return mix_reward

    def get_reward(self, compare_images, prompts=None, lambda_=0.3):
        if self.rtype == "i2i":
            reward = self.i2i_similarity_fn(compare_images)
            reward = (reward + 1) / 2
        elif self.rtype == "t2i":
            if prompts is None:
                raise ValueError("Prompts must be provided for text-to-image reward model")
            reward = self.t2i_similarity_fn(compare_images, prompts)
            reward = (reward + 1) / 2
        elif self.rtype == "mix":
            if prompts is None:
                raise ValueError("Prompts must be provided for mix-up reward model")
            reward = self.mix_up_similarity_fn(compare_images, prompts, lambda_)
        else:
            raise NotImplementedError("Unsupported reward type")
            
        return reward

def main():
    reference_dir = "../dreambooth/dataset/dog"
    compare_dir = "../dreambooth/dataset/dog6"
    start = time.time()
    print("-------------------- DINO Score -----------------------")
    similarity = DINO_score(reference_dir, reference_dir)
    print(f"Simalarity between the same images: {similarity:.4f}")
    similarity = DINO_score(reference_dir, compare_dir)
    print(f"Simalarity between different images: {similarity:.4f}")

    print("-------------------- CLIP-I Score -----------------------")
    similarity = CLIP_I_score(reference_dir, reference_dir)
    print(f"Simalarity between the same images: {similarity:.4f}")
    similarity = CLIP_I_score(reference_dir, compare_dir)
    print(f"Simalarity between different images: {similarity:.4f}")

    print("-------------------- CLIP-T Score -----------------------")
    prompts = ["a photo of a dog"]
    similarity = CLIP_T_score(prompts, reference_dir)
    print(f"CLIP-T score of prompts and images: {similarity:.4f}")

    print("-------------------- Reward Model -----------------------")
    reward_model = RewardModel(reference_dir, rtype="mix")
    images = read_images(reference_dir)
    prompts = ["a photo of dog"] * 5
    compare_images = read_images(compare_dir)
    same_reward = reward_model.get_reward(images, prompts)
    different_reward = reward_model.get_reward(compare_images, prompts)
    end = time.time()
    print(f"average reward for the same dog: {np.mean(same_reward):.4f}")
    print(f"average reward for different dogs: {np.mean(different_reward):.4f}")
    print(f"Time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    pass
    # main()