import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()