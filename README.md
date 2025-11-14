# Dogs vs Cats Image Classification Project

A comprehensive machine learning project implementing CNN and AlexNet architectures for binary and multi-class image classification tasks.

## Project Overview

This project was developed as part of the IE4483 Artificial Intelligence and Data Mining course at Nanyang Technological University (Academic Year 2025/2026, Semester 1). The project explores image classification using deep learning techniques, comparing custom CNN architectures with transfer learning approaches.

## Team Members

- **Pagdanganan Robert Martin Gosioco** 
- **Tan Chuan Bing** 
- **Tan Hong Yu** 

School of Electrical and Electronic Engineering, Nanyang Technological University

## Project Structure

### Part 1: Dogs vs Cats Binary Classification

#### Dataset
- **Training Set**: 10,000 cat images + 10,000 dog images
- **Validation Set**: 2,500 images per class
- **Test Set**: 500 unlabeled images
- **Image Resolution**: 150×150 pixels (resized from original)

#### Data Preprocessing
- Image resizing to 150×150 pixels
- Pixel normalization (scaled by 1/255 to [0,1] range)
- Data augmentation techniques:
  - Random rotation (±40°)
  - Width and height shifts (±20%)
  - Random shearing (±20%)
  - Random zooming (±20%)
  - Horizontal flipping

### Part 2: CIFAR-10 Multi-Class Classification

- **Dataset**: CIFAR-10 (10 classes, 32×32 images)
- **Approach**: Transfer learning with AlexNet
- **Image upscaling**: 32×32 → 224×224 pixels
- **Additional augmentation**: ColorJitter for brightness, contrast, and saturation

## Model Architectures

### Custom CNN Model

**Architecture Summary:**
- 4 Convolutional layers (Conv2D)
- 4 Max Pooling layers
- Flatten layer
- 3 Dense layers (128, 84, 1 neurons)
- Dropout layer (rate: 0.5)
- **Total Parameters**: 2,005,097 (7.65 MB)

**Key Features:**
- Activation: ReLU (hidden layers), Sigmoid (output)
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (default learning rate: 0.001)
- Training: 20 epochs, batch size 32
- Random seed: 10

**Performance:**
- Validation Accuracy: ~87%

### AlexNet Transfer Learning Model

**Architecture:**
- Pre-trained AlexNet with ImageNet weights
- Modified final fully connected layer
- Output: 2 classes (Dogs vs Cats) or 10 classes (CIFAR-10)

**Key Features:**
- Optimizer: AdamW (learning rate: 3×10⁻⁴)
- Loss Function: Cross-Entropy Loss
- Learning Rate Schedule: Cosine Annealing
- Training: 30 epochs, batch size 32
- Weight decay: 1×10⁻⁴
- Mixed precision training enabled
- Random seed: 42

**Performance:**
- Dogs vs Cats Validation Accuracy: 95.6%
- CIFAR-10 Test Accuracy: 82.87%

## Implementation Details

### Requirements
- Python 3.x
- TensorFlow/Keras (CNN model)
- PyTorch (AlexNet model)
- torchvision
- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)

### Running the Code

#### CNN Model (Jupyter Notebook)
1. Open `CNN_train_n_test.ipynb` in Jupyter Notebook or Google Colab
2. Follow the numbered markdown cells in sequence
3. Ensure dataset path is correctly set
4. Run cells sequentially for training and testing

#### AlexNet Model (Python Script)
```bash
# Training
python alexnet_train.py

# Testing
python alexnet_test.py
```

### Data Handling Techniques

**Class Imbalance (CIFAR-10 with Imbalanced Data):**
1. **WeightedRandomSampler**: Oversamples minority classes during training
2. **Class-Balanced Loss**: Reweights loss contribution by effective number of samples (β≈0.999)

## Results

### Dogs vs Cats Classification

| Model | Validation Accuracy | Notable Characteristics |
|-------|-------------------|------------------------|
| Custom CNN | 87% | Trained from scratch, 20 epochs |
| AlexNet | 95.6% | Transfer learning, 30 epochs |

**Confusion Matrix (AlexNet):**
- 56 cats misclassified as dogs
- 54 dogs misclassified as cats

### CIFAR-10 Classification

**Per-Class Performance (F1-Score):**
- Airplane: 0.8270
- Automobile: 0.9260
- Bird: 0.7245
- Cat: 0.7197
- Deer: 0.7883
- Dog: 0.7751
- Frog: 0.8788
- Horse: 0.8475
- Ship: 0.9082
- Truck: 0.8819

## Key Findings

1. **Model Comparison**: Transfer learning with AlexNet significantly outperforms custom CNN (95.6% vs 87%), demonstrating the value of pre-trained features.

2. **Architecture Impact**: 
   - Deeper networks (AlexNet) capture more complex patterns
   - ReLU activation mitigates vanishing gradient issues
   - Dropout and weight decay effectively prevent overfitting

3. **Image Quality**: Both models perform well on high-resolution, clear images but struggle with:
   - Low-resolution silhouettes
   - Images lacking texture details
   - Cluttered or ambiguous backgrounds

4. **Data Augmentation**: Essential for improving generalization and preventing overfitting on limited datasets.

## Reproducibility

Both models use fixed random seeds to ensure reproducibility:
- **CNN Model**: seed = 10
- **AlexNet Model**: seed = 42

For deterministic behavior in PyTorch (AlexNet):
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## File Structure

```
project/
├── datasets/
│   ├── train/
│   ├── val/
│   └── test/
├── CNN_train_n_test.ipynb
├── alexnet_train.py
├── alexnet_test.py
├── predictions.csv
├── alexnet_submission.csv
├── checkpoints/
│   └── best_model.pt
└── README.md
```

## Source Code

Complete source code and project materials are available at:
https://github.com/pagdan/NTU_IE4483_Y4S1

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Dogs vs Cats Dataset: Provided via NTULearn

## License

This project was completed as part of academic coursework at Nanyang Technological University. All rights reserved.

---

**Course**: IE4483/EE6483 Artificial Intelligence and Data Mining  
**Institution**: School of Electrical and Electronic Engineering, NTU  
**Semester**: Academic Year 2025/2026, Semester 1  
**Submission Date**: November 2025
