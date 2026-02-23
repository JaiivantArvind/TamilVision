# TamilVision 247

**Offline/Online Tamil Character Recognition System**

## 🎯 Objective

TamilVision 247 is a high-performance deep learning system designed for recognizing Tamil characters across **247 classes**. The model supports both offline (image-based) and online (real-time) character recognition, making it suitable for document digitization, handwriting recognition, and assistive technologies.

## 🏗️ Architecture

- **Backbone**: MobileNetV3-Small
- **Attention Mechanism**: Squeeze-and-Excitation (SE) Blocks
- **Input Format**: 128x128 Grayscale images
- **Output**: 247 Tamil character classes

### Why This Architecture?

- **MobileNetV3-Small**: Optimized for edge devices and resource-constrained environments while maintaining high accuracy
- **SE Attention**: Enhances feature discrimination by adaptively recalibrating channel-wise feature responses
- **Grayscale Input**: Reduces computational overhead while preserving character structure information

## 📋 Requirements

- Python 3.8+
- CUDA 11.8 (GTX 1650 compatible)
- 4GB+ GPU Memory recommended

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd T-2

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

The model is trained on **247 Tamil character classes** including:
- Vowels (உயிரெழுத்துகள்)
- Consonants (மெய்யெழுத்துகள்)
- Compound characters (உயிர்மெய் எழுத்துகள்)

## 🔧 Model Specifications

| Component | Specification |
|-----------|---------------|
| Input Size | 128x128x1 (Grayscale) |
| Architecture | MobileNetV3-Small + SE Attention |
| Parameters | ~1.5M (lightweight) |
| Output Classes | 247 |
| Framework | PyTorch 2.1.0 |

## 💻 Usage

### Training

```python
# Training code coming soon
python train.py --config config.yaml
```

### Inference

```python
# Inference code coming soon
python predict.py --image path/to/tamil_char.png
```

### API Server

```bash
# Start FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Performance

- **Target Accuracy**: >95% on test set
- **Inference Speed**: <50ms per image (GPU)
- **Model Size**: ~6MB (quantized)

## 🛠️ Hardware Compatibility

Optimized for:
- NVIDIA GTX 1650 (4GB VRAM)
- CUDA 11.8
- Edge devices with limited computational resources

## 📝 License

[Specify License]

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

[Your Contact Information]

---

**Built with ❤️ for Tamil Language Preservation**
