# Pixel Accurate Hologram Placement with AR Devices

This project aims to enhance the user experience when using AR devices such as the Microsoft HoloLens 2. Our main target is accurately capturing the user intent as to where a hologram should be placed, as well as minimizing drift of the hologram itself over varying conditions.

## Installation

Using an environment manager is highly recommended. We used [miniconda](https://docs.anaconda.com/free/miniconda/) and provide the used environment file *holoproject.yml*.

## Data

All relevant 3D data and the images with respective intrinsic and extrinsic matrices are provided in the folder */scan*.

## Outputs

Additional relevant screenshots which we collected during the course of this project are provided in the folder */outputs*

## 3D Hologram Models

Placing holograms in an environment requires a 3D model of the hologram. We opted to use a dice as this makes it easy to identify rotation and placement of the hologram. The model we used is provided in the folder */hologram_models*

## Scripts

The source code we developed is provided in the folder */scripts*. Firstly, *webcam_BRISK_record_telemetry.py* was used to evalutate BRISK in general. It provides general telemetry data of the algorithm and can record the tracked keypoints. In contrast to our final results, here we experimented with optical flow tracking of selected keypoints.

The file *combined.py* was our main testing code. Upon execution the user is presented with the specified image and several highlighted keypoints. When selecting on of the keypoints by clicking on the image, an instance of Open3D opens and shows the pointcloud from the perspective of the image. The rays indicating the x, y and z axis of the camera world coordinates are shown. Furthermore, a ray pointing in direction of the camera view and the main interest, a ray pointing towards the selected keypoint are also shown. The user can then rotate and move the view to get a better understanding of the camera position and the rays.