# Fashion MNIST Deep Learning: CNN Architecture Comparison

## 🎯 Business Impact
**Objective:** Automate e-commerce product categorization to reduce manual labeling costs by 60-70% through computer vision automation.

## 📊 Project Overview
Systematic comparison of three CNN architectures for fashion image classification, evaluating accuracy vs. complexity tradeoffs for production deployment.

| Model | Val Accuracy | Parameters | Key Insight |
|-------|-------------|------------|-------------|
| Custom Simple CNN | 91.8% | 1.2M | Efficient baseline |
| **Custom Deeper CNN** | **91.9%** | **3.8M** | **Best tradeoff (recommended)** |
| VGG16 (Transfer Learning) | 87.0% | 14.7M | Struggles with low-resolution inputs |

## 🛠️ Technical Implementation

### **Key Features**
- **Architecture Comparison:** Simple CNN vs. Deeper CNN vs. VGG16 transfer learning
- **Performance Optimization:** Mixed precision training + XLA compilation (4.8x speedup)
- **Production Pipeline:** TensorFlow Dataset API with prefetching & GPU optimization
- **Model Analysis:** Comprehensive evaluation (accuracy, confusion matrix, inference speed)

### **Technical Stack**
- **Frameworks:** TensorFlow 2.x, Keras
- **Optimization:** Mixed precision training, XLA compilation
- **Visualization:** Matplotlib, Seaborn, scikit-learn metrics
- **Infrastructure:** GPU-accelerated training (Colab T4 GPU)

## 📈 Key Findings

### **1. Custom Architectures Outperform Transfer Learning**
- VGG16 achieved only 87% accuracy vs. 91.9% for custom CNN
- **Insight:** Pre-trained models struggle with low-resolution (28×28) specialized domains
- **Recommendation:** Custom CNNs are superior for production e-commerce applications

### **2. Optimal Architecture: Deeper CNN**
- **Accuracy:** 91.9% validation accuracy
- **Efficiency:** 3.8M parameters → <10ms inference time
- **Generalization:** Minimal overfitting (1.8% train-val gap)

### **3. Production Considerations**
- **Inference Speed:** <10ms per image on T4 GPU
- **Scalability:** Optimized data pipeline handles batch processing
- **Maintenance:** Model monitoring recommendations included

## 🔗 Access the Project

### **Option 1: Interactive Colab Notebook (Recommended)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oIF4q5sabWbwMSLF0H6AOS2iUjSUr3CN?usp=sharing)

Click the badge above to open the **complete interactive notebook** with all code, visualizations, and outputs in Google Colab.

### **Option 2: Python Script**
If you prefer to run locally or review the code:
```bash
python fashion_mnist_cnn.py


## 🚀 Getting Started

### **Interactive Exploration (Quickest)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oIF4q5sabWbwMSLF0H6AOS2iUjSUr3CN?usp=sharing)

### **Local Development**
```bash
# Clone repository
git clone https://github.com/SWMLearner/deep-learning-fashion-classifier.git
cd deep-learning-fashion-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Python script
python fashion_mnist_cnn.py
