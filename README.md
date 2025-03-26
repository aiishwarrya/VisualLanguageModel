# Visual Search Using Vision-Language Models  

![image](https://github.com/aiishwarrya/VisualLanguageModel/blob/main/ss/image.png)

---

## Introduction  

This project is under the **Intel® Unnati Industrial Training 2025** initiative and is developed by **Team Raven**, consisting of **Aishwarya Joshi** and **Mahathi R**.  

In an era where information retrieval is dominated by text-based search engines, visual search is emerging as a transformative technology. Traditional search methods rely heavily on keyword matching, often failing to capture the **semantic intent** behind a query. This project explores **Vision-Language Models (VLMs)** to bridge this gap, enabling a more intuitive and **context-aware** search experience.  

We are building a **visual search engine** powered by a **custom Vision-Language Model** that prioritizes **SigLip over CLIP** for improved contrastive learning. Our approach focuses on **contrastive learning, vision encoding, and text generation** to create a **highly efficient image-text retrieval system**. SigLip enhances **training stability, retrieval accuracy, and computational efficiency**, making searches more robust and reliable.  

### Key Features  

- **Text-to-Image Search**: Retrieve images based on natural language descriptions.  
- **Image-to-Image Search**: Find visually similar images from a dataset.  
- **Hybrid Querying**: Combine text and images for enhanced search precision.  
- **Contextual Understanding**: Moves beyond keyword dependency to capture **semantic meaning**.  
- **Efficient Indexing**: Utilizes optimized retrieval techniques for large-scale datasets.  

## Objectives  

This project aims to develop a **Visual Search Engine** using a **custom-built VLM**, leveraging **contrastive learning, multi-head attention, and transformer-based architectures** to align images and text in a **shared semantic space**. The core objectives include:  

### 1️⃣ Implementing Contrastive Learning for Vision-Language Alignment  
- Prioritizing **SigLip over CLIP** due to its improved numerical stability and better softmax behavior, preventing gradient explosion during contrastive training.  
- Training the model on paired **image-text datasets**, ensuring robust **cross-modal embeddings**.  

### 2️⃣ Developing a High-Performance Vision Encoder  
- Use a **Vision Transformer (ViT)** backbone for **image feature extraction**, incorporating **patch embeddings, positional encodings, and self-attention mechanisms**.  
- Optimize training stability with **Batch Normalization, Layer Normalization, and RMS Normalization**, ensuring efficient gradient flow and weight updates.  

### 3️⃣ Implementing an Efficient Text Encoder  
- Fine-tune a **decoder-only language model** to generate high-dimensional embeddings for textual queries.  
- Employ **Rotary Positional Embeddings (RoPE)** for improved contextual understanding in transformer-based sequence processing.  

### 4️⃣ Optimizing Inference with KV-Cache & Attention Mechanisms  
- Implement **Key-Value (KV) Caching** to store previously computed attention scores, reducing redundant computations and accelerating inference.  
- Utilize **Grouped Query Attention (GQA)** to improve memory efficiency and speed during large-scale query processing.  

### 5️⃣ Enhancing Image-Text Retrieval Efficiency  
- Implement **Image Features Projection** to align vision embeddings with textual embeddings in a **latent space**.  
- Use **Top-P (Nucleus) Sampling and Temperature Scaling** during inference to control the diversity and randomness of text-based retrieval.  

### 6️⃣ Deploying and Benchmarking Performance  
- Evaluate retrieval accuracy using standard **vision-language benchmarks** and compare results against state-of-the-art **VLM models**.  
- Fine-tune indexing and retrieval techniques for **scalability** in large datasets.

---

## Approach taken and Why?

To build a **Visual-Language Model (VLM)** that understands both text and images, we use **contrastive learning**—a method that trains the model to pull matching image-text pairs closer while pushing mismatched ones apart. This allows the model to build a **shared semantic space** where similar concepts are aligned, even if they come from different modalities.

### Why Contrastive Learning?  
Traditional supervised learning struggles to capture the **relationships** between text and images since they exist in different representational spaces. Contrastive learning solves this by ensuring:  

- **Semantic Alignment**: Text and images with similar meanings have closer embeddings.  
- **Zero-Shot Learning Capability**: Once trained on diverse data, the model generalizes to unseen text-image pairs without retraining.  
- **Better Representation Learning**: Unlike simple classification, contrastive learning teaches the model to understand nuanced relationships.  

### Implementation in Our Model  
Instead of using **CLIP**, which has known issues with numerical stability in the **Softmax function**, we implement **SigLip**, an improved contrastive learning technique.  

- **SigLip optimizes Softmax behavior**, preventing gradient explosion and ensuring stable training.  
- **We train on large-scale image-text pairs**, embedding them in a unified space for robust retrieval.  
- **The model is optimized using normalized temperature scaling**, improving convergence speed and accuracy.

With a well-trained contrastive learning backbone, our VLM can match queries with the most relevant images, making visual search highly effective.  

---

## **Vision Transformer (ViT): How Our Model Sees Images**  

To make sense of images, our Vision-Language Model (VLM) needs a **Vision Transformer (ViT)**—a deep learning architecture that encodes images into numerical representations. Instead of analyzing images pixel by pixel, **ViT divides them into smaller patches and processes them like a sequence**, similar to how transformers handle text.  

### **Why Vision Transformers?**  
Unlike CNNs (Convolutional Neural Networks), which focus on local patterns, ViTs learn **global relationships** within an image using **self-attention mechanisms**. This allows our model to:  
✅ Capture **long-range dependencies** between different parts of an image.  
✅ Learn **contextual relationships** instead of just edges, textures, or colors.  
✅ Align image representations with text embeddings for **better contrastive learning**.  

### **How It Works in Our VLM**  
1️. **Image Patching** → The input image is split into fixed-size patches (e.g., 16×16 pixels).  
2️. **Linear Embedding** → Each patch is converted into a numerical vector using a learnable linear projection.  
3️. **Position Encoding** → Since transformers don’t inherently understand spatial relationships, we add **Rotary Positional Encoding (RoPE)** to retain the structure of the image.  
4️. **Self-Attention** → The model applies **multi-head self-attention**, allowing it to compare patches and **understand their relationships** within the image.  
5️. **Final Image Representation** → The output is a **compressed numerical representation of the image**, which can now be aligned with text using **contrastive learning**.  

## **Contrastive Learning: Why We Use SigLip Instead of CLIP**  
Our VLM trains its vision encoder using **contrastive learning**, where paired **image-text** data is used to bring matching pairs closer in embedding space while pushing non-matching pairs apart.  
- CLIP originally introduced contrastive learning for vision-language tasks.  
- **We use SigLip instead of CLIP** because it improves the **numerical stability** of the Softmax function, preventing gradient explosion and leading to more **robust training**.  
- SigLip enhances the alignment of **image and text embeddings**, making visual search more **accurate and efficient**.  

### **Implementation Breakdown**  
- **Coding SigLip’s Vision Encoder** → We implement the **ViT-based encoder** optimized for contrastive learning.  
- **Applying Normalization** → We integrate **BatchNorm, LayerNorm, and RMSNorm** to ensure stable training.  
- **Using RoPE** → We apply **Rotary Positional Encoding** to help the model understand spatial relationships within the image.  
