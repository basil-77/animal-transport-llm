# Animal Transport Planning (Baseline)

Baseline implementation of an animal transportation planning system.

The system determines possible transport methods and estimated travel time based on:

- an image of an animal
- origin city
- destination city

The baseline system combines computer vision, geographical calculations and rule-based reasoning.

---

# Problem

Given:

- a photo of an animal
- origin location
- destination location

the system must determine:

- possible transport modes
- estimated travel time for each transport mode

---

# Architecture

The baseline system is implemented as a modular pipeline consisting of several components.

Pipeline:
Image → Perception Agent → Geo Agent → Routing Agent → Policy Agent → JSON result

### Components

#### Perception Agent

Determines the animal class from an image.

Uses:

- **CLIP (ViT-B/32)**

Example classes:
dog
cat
bird
cow
horse
spider
butterfly


---

#### Geo Agent

Computes geographical properties:

- coordinates of cities
- distance between cities (Haversine formula)
- country
- continent

Libraries:

- `geopy`

---

#### Routing Agent

Estimates travel time using average transport speeds.

Transport speeds:

| Transport | Speed |
|--------|--------|
| Car | 70 km/h |
| Train | 120 km/h |
| Plane | 800 km/h |

ETA formula:
time = distance / speed


---

#### Policy Agent

Determines allowed transport modes using rule-based reasoning and LLM.

Uses:

- **Mistral-7B-Instruct**
- **Sentence Transformers**
- **FAISS**

Pipeline:

1. Retrieve transport rules using embeddings
2. Send retrieved rules to LLM
3. Generate structured JSON output

---

# Installation

Clone repository:
git clone https://github.com/basil-77/animal-transport-llm

cd animal-transport-llm


Install dependencies:
pip install -r requirements.txt


---

# Usage

Run the baseline system:
python main.py <image_path> <origin_city> <destination_city>


Example:
python main.py "test.jpg" Berlin Tokyo


Example output:
{
"physical_profile": {
"type": "dog"
},
"geo": {
"distance_km": 8921
},
"transport_options": {
"plane": {
"allowed": true,
"eta_hours": 14.2
}
}
}


---

# Dependencies

Main libraries:

- PyTorch
- open-clip
- sentence-transformers
- FAISS
- geopy
- vLLM

---

# Limitations

The baseline implementation contains several simplifications:

- transport times are estimated using constant speeds
- routing does not consider real schedules
- legal transport restrictions are not included
- animal classification is limited to a small set of classes

---

# Repository

Source code is available at:

https://github.com/basil-77/animal-transport-llm








