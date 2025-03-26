# Visual Search Using PaliGemma VLM  

## Introduction  

This project is under the **Intel® Unnati Industrial Training 2025** initiative and is developed by **Team Raven**, consisting of **Aishwarya Joshi** and **Mahathi R**  

In an era where information retrieval is dominated by text-based search engines, visual search is emerging as a transformative technology. Traditional search methods rely heavily on keyword matching, often failing to capture the **semantic intent** behind a query. This project explores **Vision-Language Models (VLMs)** to bridge this gap, enabling a more intuitive and **context-aware** search experience.  

We are building a **visual search engine** powered by **PaliGemma**, a state-of-the-art **VLM** that aligns visual and textual representations in a shared embedding space. By leveraging contrastive learning and advanced deep learning architectures, this model enables users to perform searches using both **text and images**, making retrieval significantly more accurate and flexible.  

## Key Features  

- **Text-to-Image Search**: Retrieve images based on natural language descriptions.  
- **Image-to-Image Search**: Find visually similar images from a dataset.  
- **Hybrid Querying**: Combine text and images for enhanced search precision.  
- **Contextual Understanding**: Moves beyond keyword dependency to capture **semantic meaning**.  
- **Efficient Indexing**: Utilizes optimized retrieval techniques for large-scale datasets.

## Why Visual Search?  

Traditional search engines struggle with queries that require **contextual reasoning** beyond simple keyword matches. Visual search, powered by **contrastive learning** and **transformers**, allows AI models to **understand** concepts rather than just matching strings. This makes it highly effective in applications like:  

- **E-Commerce**: Search for products using photos instead of text.  
- **Medical Imaging**: Retrieve similar medical scans for diagnosis.  
- **Creative Design**: Find visually related artworks, styles, or concepts.  

This project aims to implement and document **every step** of building a **fully functional visual search engine**, exploring the theoretical foundations and practical implementation of **PaliGemma** in detail.  

## Objectives  

This project aims to develop a **Visual Search Engine** using **PaliGemma**, leveraging **contrastive learning, multi-head attention, and transformer-based architectures** to align images and text in a **shared semantic space**. The core objectives include:  

### 1️⃣ Implementing Contrastive Learning for Vision-Language Alignment  
- Utilize **CLIP-style contrastive learning** to train the model on paired **image-text datasets**, ensuring robust **cross-modal embeddings**.  
- Explore **SigLip**, a recent improvement over CLIP, which enhances **numerical stability** in the **Softmax function** to prevent gradient explosion during contrastive training.  

### 2️⃣ Developing a High-Performance Vision Encoder  
- Use a **Vision Transformer (ViT)** backbone for **image feature extraction**, incorporating **patch embeddings, positional encodings, and self-attention mechanisms**.  
- Optimize training stability with **Batch Normalization, Layer Normalization, and RMS Normalization**, ensuring efficient gradient flow and weight updates.  

### 3️⃣ Implementing an Efficient Text Encoder  
- Fine-tune the **Gemma language model** to generate high-dimensional embeddings for textual queries.  
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

## Why PaliGemma?  

PaliGemma is a **Vision-Language Model (VLM)** designed to improve how AI processes and understands images and text together. It builds on existing models like CLIP but addresses their limitations by improving **contrastive learning, vision encoding, and text generation**.  

The model consists of two main parts:  
- **A Vision Transformer (ViT):** Extracts features from images efficiently.  
- **A decoder-only language model (Gemma):** Generates and understands text based on visual inputs.  

One of its key innovations is **SigLip**, which enhances the **contrastive learning process** by stabilizing softmax-based similarity calculations. This leads to **better training stability** and improved image-text matching.  

Additionally, **Grouped Query Attention (GQA)** reduces computational overhead while maintaining accuracy, and **KV-Cache** enables efficient inference by reusing past computations instead of recalculating them for every new token.  

### Key Advantages  

- **SigLip for Stability** - Enhances contrastive learning by improving softmax behavior, leading to more stable training and better image-text alignment.  

- **Efficient Attention Mechanisms** - Uses Grouped Query Attention (GQA) to speed up computations and reduce memory requirements while maintaining accuracy.  

- **Optimized Inference with KV-Cache** - Stores key-value pairs to avoid redundant calculations, making inference faster and more efficient.  

PaliGemma is designed to be **lightweight, scalable, and efficient**, making it practical for real-world applications where traditional VLMs would require extensive computational resources.  



