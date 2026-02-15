# Q-CNN Speech Command Recognition

This repository contains a **Quantized Convolutional Neural Network (Q-CNN)** for recognizing speech commands. The model is trained on a subset of the [Google Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) and quantized to **TFLite INT8** for efficient inference on edge devices.

---

## Features

- Supports **3 commands**: `on`, `happy`, `follow` (extendable to more commands).  
- Converts audio `.wav` files to **Mel-spectrograms** for CNN input.  
- Model is **quantized** for faster and smaller deployment.  
- Includes an **example Colab notebook** to demonstrate training, conversion, and inference.

---

## Repository Contents
├── qcnn_model.tflite # Quantized TFLite model
├── train_and_convert.ipynb # Colab notebook with training & conversion steps
├── preprocess.py # Optional: preprocessing scripts
├── requirements.txt # Required Python packages
└── README.md


---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/qcnn-speech-commands.git
cd qcnn-speech-commands
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Requirements include: tensorflow, librosa, numpy, matplotlib.

Usage
Using the Notebook

Open train_and_convert.ipynb in Google Colab or Jupyter Notebook to:

Preprocess audio files into Mel-spectrograms.

Train the CNN model.

Convert the model to quantized TFLite.

Run inference on test samples.

Using the TFLite Model
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="qcnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example: prepare one sample (64x32 Mel-spec, 1 channel)
sample = np.random.randint(-128, 127, size=(1, 64, 32, 1), dtype=np.int8)

# Run inference
interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output)
Notes

The model input is quantized INT8, so ensure the input data matches the quantization range.

The notebook provides raw and dequantized outputs for interpretation.

Extend the dataset by adding more folders of commands to improve accuracy.

References

TensorFlow Lite Model Optimization

Google Speech Commands Dataset

License

MIT License – free to use and modify.
