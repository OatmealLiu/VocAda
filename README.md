# Test-Time Vocabulary Adaptation for Language-Driven Object Detection

**ICIP, 2025**

[![Code](https://img.shields.io/badge/🌐-Project%20Page-3c78d8?style=flat-square)](https://github.com/OatmealLiu/VocAda)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square)](https://arxiv.org/abs/2506.00333)

---

<div align="left" style="margin: 16px 0;">

**Mingxuan Liu<sup>1,\*</sup>, Tyler L. Hayes<sup>2</sup>, Massimiliano Mancini<sup>1</sup>, Elisa Ricci<sup>1,3</sup>, Riccardo Volpi<sup>4</sup>Gabriela Csurka<sup>2</sup>** (*Corresponding Author)

<sub><sup>1</sup>University of Trento &nbsp;&nbsp; 
<sup>2</sup>NAVER LABS Europe &nbsp;&nbsp; 
<sup>3</sup>Fondazione Bruno Kessler &nbsp;&nbsp; 
<sup>4</sup>Arsenale Bioyards</sub>

</div>

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.9+

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/OatmealLiu/VocAda.git
cd VocAda
```

2. Create and activate conda environment:
```bash
conda create -n spotdet python=3.8
conda activate spotdet
```

3. Install PyTorch and Detectron2:
```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

4. Install other dependencies:
```bash
pip install -r req.txt
```

5. Install additional dependencies:
```bash
pip install ipython wandb einops mss opencv-python timm dataclasses ftfy regex fasttext scikit-learn lvis nltk Pillow datasets openai tenacity sentence-transformers
pip install git+https://github.com/openai/CLIP.git
```

### Dataset Setup

1. Download and organize your datasets in the `datasets/` directory
2. Update dataset paths in configuration files as needed
3. For COCO dataset, ensure the following structure:
```
datasets/
├── coco/
│   ├── val2017/
│   └── zero-shot/
│       └── instances_val2017_all_2_oriorder.json
```

## Usage

### Stage 1: Vocabulary Adaptation Pipeline

The SpotDet framework operates in two main stages. Stage 1 handles vocabulary adaptation:

#### 1. Image Captioning
Generate captions for images using vision-language models:

```bash
python run_stage1.py \
    --query-mode "captioning" \
    --dataset-name "coco" \
    --model-path "/path/to/llava/model" \
    --image-folder "./datasets/coco/val2017" \
    --image-anno-path "./datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json" \
    --question-file "./stage1_questions/list_all_objects.jsonl" \
    --answers-folder "./stage1_answers/coco" \
    --answers-file "answered_annotations_coco" \
    --num-chunks 1 \
    --chunk-idx 1
```

#### 2. Category Proposal
Generate relevant object categories using different methods:

**Embedding-based proposal:**
```bash
python run_stage1.py \
    --query-mode "proposing" \
    --pipeline-proposing "embedding" \
    --dataset-name "coco" \
    --embedding-model-name "sbert" \
    --embedding-model-size "sbert_base" \
    --proposing-thresh 0.15
```

**LLM-based proposal:**
```bash
python run_stage1.py \
    --query-mode "proposing" \
    --pipeline-proposing "llm" \
    --dataset-name "coco" \
    --llm-model-name "gpt-3.5-turbo-0125" \
    --llm-temperature 1.0
```

**Tagging-based proposal:**
```bash
python run_stage1.py \
    --query-mode "proposing" \
    --pipeline-proposing "tagging" \
    --dataset-name "coco" \
    --tags-file "datasets/tags/coco_rampp_tags_openset.json" \
    --pretrained-rampp-path "/path/to/rampp/model"
```

### Stage 2: Merging and Synonym Generation

#### Merge Captioning Results
```bash
python run_stage1.py \
    --query-mode "merging" \
    --dataset-name "coco" \
    --answers-folder "./stage1_answers/coco" \
    --answers-file "answered_annotations_coco"
```

#### Add Synonyms to Vocabulary
```bash
python run_stage1.py \
    --query-mode "add_synonyms" \
    --dataset-name "coco" \
    --llm-model-name "gpt-3.5-turbo-0125" \
    --answers-folder "./stage1_answers/coco" \
    --answers-file "answered_annotations_coco"
```

#### Merge LLM Proposals
```bash
python run_stage1.py \
    --query-mode "merging_llm_proposals" \
    --dataset-name "coco" \
    --answers-folder "./stage1_answers/coco" \
    --answers-file "answered_annotations_coco"
```

#### Merge LLM Proposals with CLIP
```bash
python run_stage1.py \
    --query-mode "merging_llm_proposals_with_clip" \
    --dataset-name "coco" \
    --embedding-model-name "clip" \
    --embedding-model-size "ViT-L/14" \
    --proposing-thresh 0.15 \
    --answers-folder "./stage1_answers/coco" \
    --answers-file "answered_annotations_coco"
```

### SpotDet Module Components

The SpotDet framework provides several key modules for vocabulary adaptation:

- **Captioner**: Image captioning using vision-language models
- **Proposer**: Category proposal methods (embedding, LLM, tagging)
- **Add Synonyms**: Vocabulary expansion with synonyms
- **Merging**: Combining different proposal methods

### Example Scripts

The `scripts_OV-COCO/` directory contains ready-to-use scripts for different experiments:

- **Stage1-a/**: Captioning scripts
- **Stage1-b/**: Proposal generation scripts  

Example usage:
```bash
# Run captioning
bash scripts_OV-COCO/Stage1-a/a_chunk_1-10_llava_captioning.sh

# Run embedding-based proposal
bash scripts_OV-COCO/Stage1-b/b_chunk_1-10_embedding_proposing.sh
```

## Project Structure

```
SpotDet/
├── SpotDet/                    # Core SpotDet framework
│   ├── captioner.py            # Image captioning modules
│   ├── proposer/              # Category proposal methods
│   │   ├── similarity_proposer.py    # Embedding-based proposals
│   │   ├── llm_proposer.py           # LLM-based proposals
│   │   ├── tagger_proposer.py        # Tagging-based proposals
│   │   └── gtruth_proposer.py        # Ground truth proposals
│   ├── add_synonyms.py         # Synonym generation
│   └── utils.py               # Utility functions
├── datasets/                  # Dataset configurations and metadata
├── scripts_OV-COCO/          # Experiment scripts
└── run_stage1.py             # Main vocabulary adaptation pipeline
```

## Supported Datasets

- **COCO**: COCO-80 (80 classes)
- **Objects365**: Objects365 v2

## Model Requirements

### Vision-Language Models
- **LLaVA**: LLaVA-1.6-Mistral-7B or LLaVA-1.6-34B
- **CLIP**: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336

### Large Language Models
- **OpenAI GPT**: GPT-3.5-turbo, GPT-4
- **Local LLMs**: LLaMA3-8B, LLaMA3-70B

### Embedding Models
- **CLIP**: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336
- **Sentence-BERT**: sbert_mini, sbert_base, sbert_search
- **OpenAI Embeddings**: emb3_small, emb3_large

## Citation
```bibtex
@article{liu2025test,
  title={Test-time Vocabulary Adaptation for Language-driven Object Detection},
  author={Liu, Mingxuan and Hayes, Tyler L and Mancini, Massimiliano and Ricci, Elisa and Volpi, Riccardo and Csurka, Gabriela},
  journal={arXiv preprint arXiv:2506.00333},
  year={2025}
}
```