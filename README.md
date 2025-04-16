# AutoSense: Real-Time Drowsiness Detection & Emergency Alert System using Deep Learning

*AutoSense* is a real-time driver drowsiness detection system that utilizes a deep learning model based on InceptionV3 and integrates OpenCV, dlib, and a custom-trained eye state classification model to predict drowsiness and trigger emergency alerts.

---

## Features

- Real-time drowsiness detection using webcam.
- Eye detection using dlib's 68-point facial landmark detector.
- Classification of eye state using a fine-tuned InceptionV3 model.
- Emergency alert system with siren sound for drowsy state.
- Buffer-based prediction smoothing to reduce false alerts.

---

## Model

- Base Model: InceptionV3 (pre-trained on ImageNet)
- Trained on:
  - Driver Drowsiness Dataset (DDD)
  - MRL Eye Dataset
- Classification: Binary (Drowsy / Non-Drowsy)
- Final Layers: GlobalAveragePooling → Dense (ReLU) → Dropout → Dense (Sigmoid)

---

## File Structure


### Trained model download
### .h5 file download
https://drive.google.com/file/d/1am0K1lqfZq8zP9wdZrcBUrdTXjCkYQIz/view?usp=drive_link
### Shape predictor file download for dlib
https://drive.google.com/file/d/1TeCzlV7DsliFLq_b7-Bzu8wpcsn1cFdX/view?usp=drive_link
