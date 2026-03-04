# Animal Transport Planning using Vision-Language Models

This project implements a multimodal system for planning animal transportation using image understanding and geographic reasoning.

The system receives:

- an image of an animal
- origin location
- destination location

and determines:

- physical characteristics of the animal
- possible transport modes
- estimated transportation time.

The project demonstrates how **Vision-Language Models (VLMs)** can be used for applied logistics tasks.

---

# System Overview

The system consists of four main modules:
Image → Vision-Language Model → Transport Rule Engine → ETA Estimation → JSON output



---

# Vision-Language Model

The system uses:

**Qwen2-VL-7B-Instruct**

The model extracts transport-relevant physical characteristics from the image.

Predicted attributes:
size_class
weight_class
brachycephalic
needs_carrier


Example output:
{
"size_class": "small",
"weight_class": "<1kg",
"brachycephalic": false,
"needs_carrier": true
}


The model is fine-tuned using **LoRA (Low-Rank Adaptation)**.

Inference runs locally using **vLLM**.

---

# Training Dataset

The training dataset is constructed by combining multiple open datasets.

Datasets used:

- **Oxford-IIIT Pet Dataset**
- **Animals-10 Dataset**

Images are enriched with transport-related attributes using a lookup table.

Final dataset size:

| Split | Images |
|------|------|
| Train | 31,890 |
| Validation | 1,679 |

Dataset format:
{
"image": "path/to/image.jpg",
"output": {
"size_class": "small",
"weight_class": "<1kg",
"brachycephalic": false,
"needs_carrier": "yes"
}
}



---

# Model Training

Training configuration:

| Parameter | Value |
|------|------|
| Model | Qwen2-VL-7B-Instruct |
| Adaptation | LoRA |
| LoRA rank | 16 |
| Learning rate | 5e-5 |
| Epochs | 2 |
| Batch size | 2 |
| Effective batch | 8 |

Additional techniques:

- 4-bit quantization (NF4)
- gradient checkpointing
- prefix masking

Training time:

~13 hours.

---

# Transport Rules

Transport availability is determined using:

- extracted animal attributes
- geographic properties of the route

Supported transport types:

- car
- train
- plane
- sea transport

Example rules:

- car/train allowed only within the same continent
- sea transport requires coastal countries
- brachycephalic animals may have flight restrictions

---

# ETA Estimation

Estimated travel time is calculated using average transport speeds:

| Transport | Speed |
|------|------|
| Car | 70 km/h |
| Train | 120 km/h |
| Plane | 600 km/h |
| Sea | 35 km/h |

ETA:
ETA = distance / speed


---

# Running the System

Run the system from command line:
python main_vlm.py <image_path> <origin_city> <destination_city>


Example:
python main_vlm.py "test_3.jpg" Berlin Tokyo

{
"physical_profile": {...},
"geo": {...},
"transport_options": {...}
}



---

# Dependencies

Main libraries:

- PyTorch
- HuggingFace Transformers
- vLLM
- BitsAndBytes
- FAISS
- sentence-transformers
- geopy

---

# Limitations

Current limitations include:

- simplified ETA estimation
- lack of real transport schedules
- limited rule base
- dataset imbalance toward cats and dogs (Oxford Pets)

Future work will expand the dataset with more animal species and integrate real transport regulations.

---

# Repository

Source code:

https://github.com/basil-77/animal-transport-llm




