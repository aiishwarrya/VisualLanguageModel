# Visual Search Using Vision-Language Models  

![image](https://github.com/aiishwarrya/VisualLanguageModel/blob/main/ss/image.png)

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

## Why This Approach?  

Building a **custom Vision-Language Model (VLM)** allows us to optimize performance beyond existing architectures. While models like **CLIP and PaliGemma** provide strong foundations, our approach integrates **SigLip instead of CLIP**, enhancing **contrastive learning stability and efficiency**.  

The model consists of two main components:  
- **A Vision Transformer (ViT):** Extracts features from images efficiently.  
- **A decoder-only language model:** Generates and understands text based on visual inputs.  

One of the key innovations in our approach is **SigLip**, which enhances **contrastive learning** by stabilizing softmax-based similarity calculations. This results in **better training stability** and improved image-text matching.  

Additionally, **Grouped Query Attention (GQA)** reduces computational overhead while maintaining accuracy, and **KV-Cache** enables efficient inference by reusing past computations instead of recalculating them for every new token.  

###  Key Advantages  

- **SigLip for Stability** - Replaces CLIP’s contrastive learning with a more numerically stable approach, leading to more reliable training and image-text alignment.  

- **Efficient Attention Mechanisms** - Uses Grouped Query Attention (GQA) to speed up computations and reduce memory requirements while maintaining accuracy.  

- **Optimized Inference with KV-Cache** - Stores key-value pairs to avoid redundant calculations, making inference faster and more efficient.  

This custom-built **VLM** is designed to be **lightweight, scalable, and efficient**, making it practical for real-world applications where traditional VLMs would require extensive computational resources.

