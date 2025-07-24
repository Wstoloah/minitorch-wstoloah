# MiniTorch: My Deep Learning Framework Implementation

**MiniTorch** is a pedagogical reimplementation of PyTorch designed to teach the foundations of deep learning systems. This repository contains my complete implementation of the [assignments from the Cornell Tech *Machine Learning Engineering* course](https://minitorch.github.io/), based on the MiniTorch framework.

Through this project, I built a working deep learning library from scratch — including autograd, tensor ops, CUDA backends, and training pipelines for real-world tasks.

---

## 🔧 Setup Instructions

**Python Version:** 3.11+

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install -Ue .

# Verify installation
python -c "import minitorch"
```

---

## 🗂️ Repository Structure

| Directory/File  | Description                                  |
| --------------- | -------------------------------------------- |
| `minitorch/`    | Core framework: autograd, tensors, modules   |
| `tests/`        | Unit and property tests for each module      |
| `project/`      | Training and visualization scripts           |
| `README.md`     | This file                                    |
| `sentiment.txt` | Sentiment classification training logs       |
| `mnist.txt`     | MNIST training logs                          |

---

## ✅ Assignment Overview & My Work

### Module 0: Operators, Functional Programming, Modules

* Implemented scalar operators: `add`, `mul`, `neg`, `exp`, etc.
* Built higher-order functions: `map`, `zipWith`, `reduce`
* Created parameter modules using `minitorch.Module`

---

### Module 1: Scalars & Autodifferentiation

* Implemented `Scalar`, `ScalarFunction`, and `backpropagation`
* Built autodiff graph from scratch
* Trained a scalar model using custom autograd

*Training results visualized in Streamlit*

---

### Module 2: Tensors & Tensor Autograd

* Developed `Tensor` class with broadcasting and indexing
* Implemented forward/backward passes for tensor operations
* Trained neural networks using tensor-based models

---

### Module 3: FastOps & CUDA

* Rewrote tensor ops using Numba (`map`, `zip`, `reduce`)
* Implemented parallel CPU & GPU backends (CUDA)
* Built efficient `matmul` kernels with shared memory

This parallel implementation gave a 10x speedup. On a standard Colab GPU setup, the CPU gets below 2 seconds per epoch and GPU below 1 second per epoch

---

### Module 4: Convolutions & Classification

* Implemented `conv1d`, `conv2d`, `avgpool2d`, `softmax`, `dropout`
* Trained CNN (LeNet-style) on MNIST
* Built 1D ConvNet for SST2 sentiment classification

📄 Training logs:

* [`mnist.txt`](mnist.txt): MNIST validation accuracy logs
* [`sentiment.txt`](sentiment.txt): SST2 >70% validation accuracy

---

## 🧠 Key Takeaways

* Built full autograd engine (scalar + tensor)
* Designed tensor ops with broadcasting, strides, views
* Optimized backend with Numba and CUDA
* Trained custom deep networks on real data
* Gained hands-on understanding of PyTorch internals

---

## 📸 Visual Results

> To be added

---

## 🚀 Training Commands

```bash
# Scalar model training
streamlit run project/app.py -- 1

# Tensor model training
streamlit run project/app.py -- 2

# Fast backend training (GPU/CPU)
python project/run_fast_tensor.py --BACKEND gpu --DATASET split --HIDDEN 100 --RATE 0.05

# Sentiment classification
python project/run_sentiment.py

# MNIST digit classification
python project/run_mnist_multiclass.py
```

---

## 📜 License

This work is based on the educational MiniTorch framework. My implementations are released under the MIT License.
