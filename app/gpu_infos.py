"""
GPU Information Retrieval
This script retrieves the name and memory of GPUs using nvidia-smi, if installed.

--- Requirements ---

sudo apt update
sudo apt install nvidia-driver-XXX 
# Replace XXX with the appropriate driver version

"""

import subprocess
import shutil

def get_gpu_info_nvidia_smi():
    """Retrieve the name and memory of GPUs using nvidia-smi, if installed."""
    # Check if nvidia-smi is installed
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi is not installed or not in the system PATH.")
        return

    try:
        # Execute the nvidia-smi command to retrieve GPU information
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print("Error while executing nvidia-smi:", result.stderr)
            return

        # Display GPU information
        gpus = result.stdout.strip().split("\n")
        for i, gpu in enumerate(gpus):
            name, memory = gpu.split(", ")
            print(f"GPU {i}: {name}, Total memory: {memory} MB")
    except Exception as e:
        print(f"An error occurred while trying to retrieve GPU info: {e}")


get_gpu_info_nvidia_smi()
