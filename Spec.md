Project Requirement Document (PRD): Robust Deepfake Detection System

1. Project Overview & Business Requirements
    Project Name: Robust Deepfake Detection for NTIRE 2026.

Primary Objective: Develop a multi-modal deep learning system capable of detecting AI-generated images "in-the-wild" with high robustness to unknown generators and severe image degradation.

Business Use Cases:
    Automated e-KYC Defense: Preventing synthetic identity fraud.
    Disinformation Shield: Real-time flagging of manipulated news footage.

Success Metrics: * Primary Metric: AUC (Area Under Curve).
    Target: Achieve ∼85−90% AUC on unseen models (e.g., Midjourney v6, Sora, Flux).
    Maintain high performance at JPEG Quality 50.

2. Technical Requirements & Architecture

Framework: PyTorch (latest stable version).

Local Development Environment Target: Must be compatible with modern Linux distributions (e.g., Arch Linux) and support local GPU acceleration (CUDA) for initial prototyping, with the ability to scale to cloud-based VMs (e.g., Azure).

Package Management: Node.js/Web wrappers (if any are built for the frontend interface) should strictly utilize pnpm.

System Architecture: The Hybrid-Swin Network

The core system is a Multi-Modal Hybrid Network processing the image in two simultaneous domains.

3.1. 
Stream A: Spatial Domain (RGB) 

    Purpose: Extract visual artifacts and long-range structural dependencies (e.g., blending errors, facial symmetry).

Model Backbone: Swin Transformer V2.

    Input: Normalized RGB tensor of shape 3×224×224.

    Output: Spatial feature embeddings.

3.2. 
Stream B: Frequency Domain 

    Purpose: Detect mathematical inconsistencies, checkerboard patterns, and upsampling fingerprints left by generative models.

Model Backbone: F3-Net (Frequency in Face Forgery).

Input Processing: Apply Discrete Cosine Transform (DCT) to extract frequency maps from the input image before passing to the network.

    Output: Frequency feature embeddings.

3.3. 
Fusion Block 

    Mechanism: Cross-Attention Module.

Logic: Dynamically weight the reliability of Stream A vs. Stream B based on image quality.

Output: A fused feature vector passed to a final classification head yielding a binary probability (0=Real,1=Fake).

4. Data Pipeline & Training Protocol

To achieve the "in-the-wild" robustness required, the training loop must implement specific augmentation strategies.

4.1. 
The Degradation Pipeline 

The dataloader must apply a "hostile" augmentation pipeline to all clean training images dynamically:

    Compression: Apply JPEG compression with a random Quality Factor (QF) between 30 and 60.

Blur: Apply Gaussian Blur to simulate out-of-focus optics.

Scaling: Downscale images to 64×64 pixels, then upscale back to network input size to simulate severe data loss.

4.2. 
Frequency Masking (FMSI) 

    Implementation: During the forward pass of the training loop, randomly mask (zero-out) sections of the frequency spectrum.

Goal: Force the model to learn general forgery traces rather than overfitting to specific generator fingerprints.

5. Development Phases for AI Agent

    Phase 1: Pipeline Setup: Scaffold the PyTorch project, set up dataloaders for standard datasets (e.g., FaceForensics++, Celeb-DF), and build the Degradation Pipeline using Albumentations.

    Phase 2: Stream Implementation: Implement Swin V2 and F3-Net independently. Ensure the DCT preprocessing step is highly optimized.

    Phase 3: Fusion & Training Loop: Implement the Cross-Attention Fusion block. Build the training loop incorporating Frequency Masking and AUC metric logging.

    Phase 4: Evaluation Script: Create an evaluation script that strictly measures Cross-Generator Generalization and plots ROC curves.