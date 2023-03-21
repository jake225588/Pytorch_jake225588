# YOLO PyTorch Evaluation

This directory contains the evaluation of the YOLO (You Only Look Once) object detection implementation using the PyTorch deep learning framework. YOLO is a popular and efficient real-time object detection algorithm that has been widely adopted in various fields, including science, engineering, and computer vision applications.

The PyTorch-based YOLO implementation takes advantage of the flexibility and efficiency of the PyTorch framework, making it easier to train, fine-tune, and deploy the model for various tasks. This particular YOLO variant can be used for tasks such as:

- Autonomous vehicles
- Robotics
- Surveillance and security
- Industrial automation
- Image and video analysis

YOLO PyTorch is an application built using the PyTorch framework. It is neither a programming tool nor a middleware. It is an end-to-end object detection solution that can be easily integrated into other applications and pipelines.

In this evaluation, we explore the performance, accuracy, and usability of the YOLO PyTorch implementation, as well as its relevance to specific use cases in science and engineering. We also provide example code, pre-trained models, and sample datasets to help users get started with using YOLO PyTorch for their own projects.

## Installation and Setup on HPCC

Below are the step-by-step instructions for installing and setting up the YOLO PyTorch software on the High Performance Computing Cluster (HPCC). These instructions will guide you through the process of installing the required dependencies, downloading the source code, and setting up the environment.

### Prerequisites

- Access to an HPCC (NVIDIA GPU recommended)
- Python 3.6 or later installed
- A Conda or virtualenv environment is recommended

### Step 1: Clone the Repository

Clone the YOLO PyTorch repository to your local HPCC workspace:

git clone https://github.com/jake225588/Pytorch_jake225588.git

### Step 2: Create a Virtual Environment (Optional)

It is recommended to create a virtual environment using Conda or virtualenv to manage the dependencies for the YOLO PyTorch project:

Using Conda:

conda create -n yolo_pytorch_env python=3.8

conda activate yolo_pytorch_env

### Step 3: Install Dependencies

Install the required dependencies using the provided `requirements.txt` file:

pip install -r requirements.txt

### Step 4: Install PyTorch and torchvision

Install the appropriate version of PyTorch and torchvision for your system, taking into account the available GPU and CUDA version:

For CUDA 10.2
<br>pip install torch torchvision -f https://download.pytorch.org/whl/cu102/torch_stable.html

For CUDA 11.1
<br>pip install torch torchvision -f https://download.pytorch.org/whl/cu111/torch_stable.html


Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions specific to your system.


### Step 5: Run a Test
To verify the installation and setup, run a test using the provided sample images:

python detect.py --weights yolov5s.pt --source bus.jpg


### Step 6: Submit a Job to the HPCC


sbatch yolo_pytorch_job.sh(might not work beacuse the diffrent env name)

### References

https://github.com/ultralytics/yolov5
