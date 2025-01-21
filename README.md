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

