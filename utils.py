import os
from PIL import Image
import numpy as np
from google.cloud import storage
import logging
import re
from transformers import AutoProcessor, LlavaForConditionalGeneration

import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "RL for diffusion"

def cosine_similarity(x, y):
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    similarity = np.sum((x / x_norm) * (y / y_norm), axis=-1)
    return similarity

def read_images(directory):
    images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            images.append(image)
    return images

def upload_images_to_gcs(localpath, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    for filename in os.listdir(localpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_file = os.path.join(localpath, filename)
            blob_name = os.path.join(destination_blob_name, filename)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}.")

def remove_unique_token(text, token_to_remove="sks"):
    cleaned_text = re.sub(r'\s*\b' + re.escape(token_to_remove) + r'\b\s*', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def clean_subject(subject):
    cleaned_subject = re.sub(r'\d+', '', subject)
    cleaned_subject = re.sub(r'_', ' ', cleaned_subject)
    return cleaned_subject

def upload_file_to_gcs(localpath, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(localpath)
    print(f"Uploaded {localpath} to gs://{bucket_name}/{destination_blob_name}.")

def list_npy_files(bucket_name, prefix):
    """Lists all the files in the bucket that are within the specified prefix and ends with .json."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # List all objects that start with the prefix
    print(f"blobs: {blobs}")
    return [blob for blob in blobs if blob.name.endswith('.npy')]

def list_json_files(bucket_name, prefix):
    """Lists all the files in the bucket that are within the specified prefix and ends with .json."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # List all objects that start with the prefix
    print(f"blobs: {blobs}")
    return [blob for blob in blobs if blob.name.endswith('.json')]

def download_and_save(bucket_name, blob_name, download_path):
    """Downloads a file from GCS and saves it to a specified local path."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Ensure the download directory exists
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    # Download the file to a local path
    blob.download_to_filename(download_path)
    print(f"Downloaded '{blob_name}' to '{download_path}'")

def setup_logger():
    # Configure the logging system
    logging.basicConfig(
        filename='logs/error.log',          # Log file name
        filemode='a',                  # Append mode
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        level=logging.ERROR            # Capture error and above levels of log messages
    )
    logger = logging.getLogger('GlobalErrorHandler')
    return logger

def load_prompts(unique_token, subject_token, live=False):
    if not live:
        prompt_list = [
        'a {0} {1} in the jungle'.format(unique_token, subject_token),
        'a {0} {1} in the snow'.format(unique_token, subject_token),
        'a {0} {1} on the beach'.format(unique_token, subject_token),
        'a {0} {1} on a cobblestone street'.format(unique_token, subject_token),
        'a {0} {1} on top of pink fabric'.format(unique_token, subject_token),
        'a {0} {1} on top of a wooden floor'.format(unique_token, subject_token),
        'a {0} {1} with a city in the background'.format(unique_token, subject_token),
        'a {0} {1} with a mountain in the background'.format(unique_token, subject_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, subject_token),
        'a {0} {1} on top of a purple rug in a forest'.format(unique_token, subject_token),
        'a {0} {1} with a wheat field in the background'.format(unique_token, subject_token),
        'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, subject_token),
        'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, subject_token),
        'a {0} {1} floating on top of water'.format(unique_token, subject_token),
        'a {0} {1} floating in an ocean of milk'.format(unique_token, subject_token),
        'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, subject_token),
        'a {0} {1} on top of a mirror'.format(unique_token, subject_token),
        'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, subject_token),
        'a {0} {1} on top of a dirt road'.format(unique_token, subject_token),
        'a {0} {1} on top of a white rug'.format(unique_token, subject_token),
        'a red {0} {1}'.format(unique_token, subject_token),
        'a purple {0} {1}'.format(unique_token, subject_token),
        'a shiny {0} {1}'.format(unique_token, subject_token),
        'a wet {0} {1}'.format(unique_token, subject_token),
        'a cube shaped {0} {1}'.format(unique_token, subject_token)
        ]
    else:
        prompt_list = [
            'a {0} {1} in the jungle'.format(unique_token, subject_token),
            'a {0} {1} in the snow'.format(unique_token, subject_token),
            'a {0} {1} on the beach'.format(unique_token, subject_token),
            'a {0} {1} on a cobblestone street'.format(unique_token, subject_token),
            'a {0} {1} on top of pink fabric'.format(unique_token, subject_token),
            'a {0} {1} on top of a wooden floor'.format(unique_token, subject_token),
            'a {0} {1} with a city in the background'.format(unique_token, subject_token),
            'a {0} {1} with a mountain in the background'.format(unique_token, subject_token),
            'a {0} {1} with a blue house in the background'.format(unique_token, subject_token),
            'a {0} {1} on top of a purple rug in a forest'.format(unique_token, subject_token),
            'a {0} {1} wearing a red hat'.format(unique_token, subject_token),
            'a {0} {1} wearing a santa hat'.format(unique_token, subject_token),
            'a {0} {1} wearing a rainbow scarf'.format(unique_token, subject_token),
            'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, subject_token),
            'a {0} {1} in a chef outfit'.format(unique_token, subject_token),
            'a {0} {1} in a firefighter outfit'.format(unique_token, subject_token),
            'a {0} {1} in a police outfit'.format(unique_token, subject_token),
            'a {0} {1} wearing pink glasses'.format(unique_token, subject_token),
            'a {0} {1} wearing a yellow shirt'.format(unique_token, subject_token),
            'a {0} {1} in a purple wizard outfit'.format(unique_token, subject_token),
            'a red {0} {1}'.format(unique_token, subject_token),
            'a purple {0} {1}'.format(unique_token, subject_token),
            'a shiny {0} {1}'.format(unique_token, subject_token),
            'a wet {0} {1}'.format(unique_token, subject_token),
            'a cube shaped {0} {1}'.format(unique_token, subject_token)
        ]
    return prompt_list

def training_prompts(unique_token, subject_token, live=False):
    if not live:
        prompt_list = [
            f'a {unique_token} {subject_token} in Vincent Van Gogh rendition',
            f'a {unique_token} {subject_token} on the Moon',
            f'a green {unique_token} {subject_token}',
            f'a {unique_token} {subject_token} with a China Great Wall in the background',
            f'a {unique_token} {subject_token} on top of wooden roof',
            f'a {unique_token} {subject_token} with the Statue of Liberty in the background',
            f'a {unique_token} {subject_token} with a volcano in the background',
            f'a {unique_token} {subject_token} on a subway platform',
        ]
    else:
        prompt_list = [
            f'a {unique_token} {subject_token} wearing a green outfit',
            f'a {unique_token} {subject_token} in Vincent Van Gogh rendition',
            f'a {unique_token} {subject_token} with a China Great Wall in the background',
            f'a {unique_token} {subject_token} is swimming in a pool',
            f'a {unique_token} {subject_token} on a subway platform',
            f'a {unique_token} {subject_token} with the Statue of Liberty in the background',
            f'a {unique_token} {subject_token} on the Moon',
            f'a green {unique_token} {subject_token}',
        ]
    return prompt_list

def validation_prompts(unique_token, subject_token):
    prompt_list = [
        f'a {unique_token} {subject_token} in a beautiful garden',
        f'a {unique_token} {subject_token} at the beach during sunset'
    ]
    return prompt_list

def reward_prompts(unique_token, subject_token, live=False):
    if not live:
        prompt_list = [
            f'a green {unique_token} {subject_token}',
        ]
    else:
        prompt_list = [
            f'a {unique_token} {subject_token} in Vincent Van Gogh rendition',
        ]
    return prompt_list