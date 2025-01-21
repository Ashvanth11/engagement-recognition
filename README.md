# engagement-recognition
TensorFlow based implementation of a Deep Learning network that captures facial engagement of students to assess their learning engagement using CNNs in
Massive Open Online Courses (MOOC) scenarios. 
This repository provides a complete pipeline for recognizing user engagement states (Boredom, Engagement, Confusion, Frustration) using a deep learning model trained on video frame data. The project uses a convolutional neural network (CNN) for multi-output classification and includes preprocessing, dataset management, training, testing, and evaluation modules.

## Dataset
DAiSEE is a multi-label video classification dataset comprising of 9,068 video snippets captured from 112 users for recognizing the user affective states of boredom, confusion, engagement, and frustration "in the wild". The dataset has four levels of labels namely - very low, low, high, and very high for each of the affective states, which are crowd annotated and correlated with a gold standard annotation created using a team of expert psychologists.

## Features
### Data Preprocessing:

Extracts frames from videos at custom frame rates.

Saves frame paths and corresponding labels as NumPy arrays for fast processing.

### Model Architecture:

CNN-based model with multiple output heads for recognizing engagement states.

Flexible architecture supporting additional finetuning.

### Training and Evaluation:

Handles multi-label classification for engagement recognition.

Supports callbacks such as early stopping and TensorBoard logging.

### Testing:

Separate scripts for testing models on different datasets.

Evaluates and saves accuracy for each engagement state

## Setup Instructions
### Prerequisites
Python 3.8+

TensorFlow

Keras

NumPy

pandas

ffmpeg (for frame extraction)

matplotlib


## Usage
### Step 1: Extract Frames

Run the get_frames.py script to extract frames from the DAiSEE videos.

Frames are stored in the Frames directory, organized into subdirectories for training, testing, validation, and final testing.

### Step 2: Save File Paths and Labels

Generate numpy arrays for frame file paths and labels using:

python save_filepath_label.py

The generated files are saved in the npoutput directory.

### Step 3: Train the Model

Train the CNN model using model.py.

The model architecture is defined in the get_model function, and the trained model is saved in the modelpath directory.

### Step 4: Evaluate the Model

Evaluate the trained model on the test dataset using evaluate.py.

Evaluation results are saved in the Results directory.

## Scripts Explanation

### 1. get_frames.py

Extracts frames from videos at default or specified frame rates using ffmpeg.

Handles subdirectories: Train, Test, Validation, and FinalTest.

### 2. save_filepath_label.py

Maps extracted frames to corresponding labels.

Saves frame file paths and labels as numpy arrays for use in the input pipeline.

### 3. get_dataset.py

Creates TensorFlow datasets for training and testing.

Preprocesses images: resizing, normalization, and batching.

## 4. model.py

Defines a custom CNN model for multi-label classification.

Trains the model using the training dataset.

Saves the trained model.

### 5. evaluate.py

Loads the trained model and evaluates it on the test dataset.

Outputs accuracy and saves results to numpy files.

## Results

Evaluation results, including accuracy for each class, are stored in the Results directory.

## Acknowledgments

DAiSEE dataset creators

TensorFlow/Keras documentation
