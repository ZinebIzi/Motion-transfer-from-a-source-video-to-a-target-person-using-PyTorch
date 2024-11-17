# **Motion Transfer from a Source Video to a Target Person Using PyTorch**

This project implements motion transfer from a source video to a target person, leveraging **PyTorch** and inspired by the paper *"Everybody Dance Now"* (ICCV 2019) by Chan et al. It explores Generative Adversarial Networks (GANs) and other techniques to generate realistic motion transfer through a step-by-step approach, from skeleton extraction to GAN-based image generation.

---

## **Project Overview**

The main objective is to make a target person replicate the movements of a source individual by generating a frame-by-frame video. The implementation follows these steps:

1. **Skeleton Extraction**: Using the Mediapipe Pose Landmarker, skeletons (poses) are extracted from source and target videos.
2. **Image Generation**: A neural network learns to synthesize images of the target person in various poses, ultimately generating a video.

---

## **Workflow**

### **Data Preparation**
1. **Extract Video Frames**:
   - Frames are extracted from videos using the `VideoSkeleton` class.
   - Skeleton data is generated for each frame and stored for training/testing.

### **Model Stages**
1. **Closest Skeleton (Nearest Neighbor)**:
   - Matches the target skeleton with the closest one in the dataset using a nearest-neighbor approach.
   - Implemented in `GenNearest::generate`.
2. **Direct Neural Network**:
   - Trains a simple neural network (NN) to generate images from skeleton data.
   - Skeletons are represented as a reduced 2D array of 13 joints (26 numbers).
   - Implemented in `GenVanillaNN`.
3. **Skeleton-to-Image Neural Network**:
   - Converts the skeleton into a "stick" figure image (via `Skeleton::draw_reduced`) before feeding it into the network.
4. **GAN-Based Generator**:
   - Incorporates a discriminator to improve the quality of the generated images.
   - Aims to produce realistic images with enhanced temporal continuity.

---

## **Code Explanation**

### **VideoReader file**:
This script defines a `VideoReader` class that simplifies video file handling using OpenCV. It provides methods to:
   - Open and read video frames individually or in batches.
   - Retrieve video properties such as width, height, FPS, and total frame count.
   - Display video frames and control playback.

The class ensures proper resource management by releasing video files when done. It also includes an example to play a video frame-by-frame and allows quitting with the 'q' key.

---

### **Vec3 file**:
This file contains a class for 3D vector operations, leveraging **NumPy** for efficient calculations. It includes methods to:
   - Create 3D vectors and represent them as tuples.
   - Perform operations like dot product, cross product, magnitude, and power.
   - Access or set vector components using Cartesian (x, y, z), spherical, or cylindrical coordinates.
   - Seamlessly integrate with **NumPy** arrays for advanced numerical tasks.

---

### **Skeleton file**:
This script defines a `Skeleton` class for detecting, processing, and visualizing human body poses using the **MediaPipe Pose API**. It includes the following functionalities:
   
**Pose Detection**:
   - Uses MediaPipe to extract 33 3D landmarks representing key points of the human body (e.g., nose, shoulders, elbows, hips, etc.) from an input image.

**Full and Reduced Skeletons**:
   - The full skeleton includes all 33 landmarks.
   - The reduced skeleton simplifies this to 13 key joints, focusing on head, shoulders, elbows, wrists, hips, knees, and ankles.

**Data Representation**:
   - Stores each landmark as a 3D vector using the `Vec3` class for easy manipulation.
   - Converts the skeleton into a **NumPy** array for numerical operations, with optional reduction to 2D coordinates.

**Bounding Box and Cropping**:
   - Computes the bounding box around the detected skeleton.
   - Normalizes landmark positions for cropping or resizing.

**Visualization**:
   - Draws the skeleton on the input image with color-coded joints.
   - Supports both full and reduced skeleton visualizations with labeled connections between joints.

**Distance Calculation**:
   - Computes the distance between two skeletons based on their landmarks.

---

### **VideoSkeleton file**:
This script processes a video to detect and associate human skeletons with its frames using pose estimation. It extracts keypoints from video frames and saves the results for visualization and further analysis.
   - **Pose Estimation**: Detects human skeletons (keypoints) in frames using the `Skeleton` class.
   - **Frame Processing**: Processes every nth frame (configurable) to save computation, resizes images, and crops around detected skeletons.
   - **Data Storage**: Saves skeletons and corresponding frames into a file for easy reloading.
   - **Visualization**: Combines each frame with its detected skeleton and displays the results for inspection.

---

### **GenNearest file**:
This script generates new images based on skeleton posture matching using a nearest neighbor approach. Given a target skeleton, it finds the closest matching skeleton from a video and returns the corresponding image. Here's an overview:
   - **Posture Matching**: Compares a given skeleton's pose with skeletons from a reference video and selects the closest match based on a distance metric.
   - **Image Generation**: After finding the closest matching skeleton, it retrieves the image from the video that corresponds to this skeleton.
   - **Nearest Neighbor**: The closest match is determined by calculating the "distance" between the input skeleton and those in the target video.

---

### **GenVanilla file**:
This script generates new images based on skeleton postures using a neural network. The model learns to generate images either directly from skeleton data or from images with skeletons in them.

**Data Preparation**:
   - The `VideoSkeleton` object holds the skeleton and image data from a video.
   - A dataset class (`VideoSkeletonDataset`) is used to feed the skeletons and images to the neural network during training.

**Neural Network Models**:
   - `GenNNSkeToImage`: A simple feed-forward neural network that generates images from skeleton data.
   - `GenNNSkeImToImage`: A more complex model that uses both skeleton data and images to generate output images.

**Training**:
   - The `GenVanillaNN` class handles training. It uses Mean Squared Error (MSE) loss to train the network and save the model after training.
   - The model is trained using skeleton data from a video, and the goal is to generate corresponding images for each skeleton.

**Image Generation**:
   - After training, the model can generate images from new skeletons using the `generate()` function.

---

### **GenGAN file**:
This script implements a **Generative Adversarial Network (GAN)** to generate images from skeleton postures. The network consists of two main components:
   - **Generator (NetG)**: Takes a skeleton as input and generates an image based on the posture of the skeleton.
   - **Discriminator (NetD)**: Distinguishes between real and generated images, helping the generator improve over time.

**Discriminator**:
   - A convolutional neural network (CNN) classifies whether an image is real or fake.
   - It uses multiple layers of convolution and batch normalization, followed by a Sigmoid activation function to output a probability between 0 (fake) and 1 (real).

**Generator**:
   - The `GenNNSkeToImage` model generates images from skeletons using a fully connected network (or another architecture depending on the configuration).

**Training**:
   - The generator tries to create realistic images from skeleton data, while the discriminator learns to differentiate between the real and fake images.
   - The training alternates between training the discriminator and the generator, using binary cross-entropy loss.

**Image Generation**:
   - After training, the model can generate images from skeletons using the `generate()` function.

---

### **DanceDemo file**:
This file demonstrates how a dance animation from a source video is applied to a target character using a generative model.

**Set the Generator Type (GEN_TYPE)**: The `GEN_TYPE` variable defines which generative model will be used to transform the skeleton into an image. You can change the value of `GEN_TYPE` to choose a specific model:
   - `GEN_TYPE = 1`: Nearest Neighbor Model (`GenNearest`)
   - `GEN_TYPE = 2`: Vanilla Neural Network (Skeleton to Image) (`GenVanillaNN - optSkeOrImage=1`)
   - `GEN_TYPE = 3`: Vanilla Neural Network (Image with Skeleton to Image) (`GenVanillaNN - optSkeOrImage=2`)
   - `GEN_TYPE = 4`: Generative Adversarial Network (GAN) (`GenGAN`)

Change the `GEN_TYPE` in the code to select your desired model.

---

# **Motion-transfer-from-a-source-video-to-a-target-person-using-PyTorch**
