**Q-CNN Speech Command Recognition**

**Project:** A small Quantized Convolutional Neural Network (Q-CNN) for recognizing speech commands. The model is trained on a subset of the Google Speech Commands dataset and converted to a TFLite INT8 model for efficient edge inference.

**Features**
- **Commands:** Recognizes three example commands: `on`, `happy`, `follow` (easily extendable).
- **Preprocessing:** Converts `.wav` audio into Mel-spectrograms for CNN input.
- **Quantized model:** Exports a TFLite INT8 model for smaller size and faster inference.
- **Example notebook:** Includes a notebook demonstrating preprocessing, training, conversion, and inference.

**Repository Contents**
- `qcnn_model.tflite`: Quantized TFLite model.
- `train_and_convert.ipynb`: Notebook showing preprocessing, training, and conversion.
- `preprocess.py`: Helper for converting audio to Mel-spectrograms.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

**Setup**
- **Prerequisites:** Python 3.8+ and `pip`.
- **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Usage**

1. Preprocess and train (recommended via the notebook): open [train_and_convert.ipynb](train_and_convert.ipynb) in Jupyter or Colab and follow the cells to preprocess audio, train the model, and convert to TFLite.

2. Run inference with the TFLite model (example):

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="qcnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example: prepare one quantized sample (shape: 1, 64, 32, 1)
sample = np.random.randint(-128, 127, size=(1, 64, 32, 1), dtype=np.int8)

interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output)
```

**Notes**
- **Input quantization:** The model expects INT8 input. Match input preprocessing and quantization parameters produced during conversion.
- **Extend dataset:** Add more labeled folders (one per command) to improve accuracy.

**References**
- TensorFlow Lite Model Optimization: https://www.tensorflow.org/lite/performance/model_optimization
- Google Speech Commands dataset: https://www.tensorflow.org/datasets/catalog/speech_commands

**License**
- MIT License — free to use and modify.
