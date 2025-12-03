# Neuropedal 🎸

**Neuropedal** is a deep learning project that predicts guitar pedal settings from audio recordings. Using a trained **ResNet34 CNN** on Mel spectrograms, the system outputs `[drive, tone]` parameters to help guitarists achieve their desired tone without manual experimentation.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training a Model](#training-a-model)  
  - [Running Inference via GUI](#running-inference-via-gui)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

---

## Overview

Neuropedal leverages **deep learning** to analyze audio from a guitar and automatically suggest pedal settings:

- **Input:** 1-channel Mel spectrogram (from a guitar WAV file)  
- **Output:** 2-value regression `[drive, tone]`  
- **Model:** ResNet34 adapted for single-channel audio input and regression output  

This project eliminates the time-consuming trial-and-error process of adjusting pedals, providing musicians with instant tone recommendations.

---

## Features

- Trained CNN predicts `[drive, tone]` values  
- GUI for easy drag-and-drop inference using PyQt5  
- Preprocessing includes waveform padding/trimming, mono conversion, and normalized Mel spectrograms  
- Training and inference pipelines fully implemented  
- Works with any pedal compatible with `[drive, tone]` settings  

---

## Project Structure

