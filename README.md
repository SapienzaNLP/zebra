
<div align="center">
  <img src="assets/zebra.jpeg" height="80">
</div>

<div align="center">

# ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering

[![Conference](https://img.shields.io/badge/EMNLP-2024-4b44ce)](https://2024.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-placeholder-b31b1b.svg)](https://arxiv.org/abs/placeholder)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/sapienzanlp/zebra-66e3ec50c8ce415ea7572d0e)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
</div>

A retrieval augmnetation framework for zero-shot commonsense question answering with LLMs. 

## 🛠️ Installation

Installation from PyPi

```bash
pip install zebra
```

Installation from source

```bash
git clone https://github.com/framolfese/zebra.git
cd zebra
conda create -n zebra python==3.10
conda activate zebra
pip install -e .
```

## 🚀 Quick Start

ZEBRA is a plug-and-play retrieval augmentation framework for **Commonsense Question Answering**. \
It is composed of two pipeline stages: *knowledge generation* and *informed reasoning*. \
The knowledge generation step is responsible for:
- retrieving relevant examples of question-knowledge pairs from a large collection
- prompting a LLM to generate useful explanations for the given input question by leveraging the relation between the retrieved question-knowledge pairs.

The informed reasoning step is responsible for prompting a LLM for the question answering task by taking advantage of the previously generated explanations.

Here is an example on how you can use ZEBRA for Question Answering:

```python
from zebra import Zebra

# Load Zebra with language model, retriever, document index and explanations.
zebra = Zebra(model="meta-llama/Meta-Llama-3-8B-Instruct")

# Provide a question and answer choices.
questions = [
    "What should you do if you see someone hurt and in need of help?",
    "If your friend is upset, what is the best way to support them?",
    "What should you do if your phone battery is running low in a public place?",
    "What should you do if you are running late for an important meeting?",
]

choices = [
    ["Walk away.", "Call for help.", "Take a photo for social media."],
    ["Listen to them and offer comfort.", "Tell them they are overreacting.", "Ignore them and walk away."],
    ["Borrow a stranger's phone.", "Use public charging station.", "Leave your phone unattended while it charges."],
    ["Rush through traffic.", "Call and inform them you will be late.", "Do not show up at all."],
]

# Generate knowledge and perform question answering.
zebra_output = zebra.pipeline(questions=questions, choices=choices)
```

Output:
```bash
  ZebraOutput(
    explanations=[
      [
        "Walking away would be neglecting the person's need for help and potentially putting them in danger.",
        'Calling for help, such as 911, is the most effective way to get the person the assistance they need.',
        "Taking a photo for social media might spread awareness, but it's not a direct way to help the person in need."
      ],
      [
        'Listening and offering comfort shows empathy and understanding.', 
        "Telling someone they're overreacting can be dismissive and unhelpful.", 
        'Ignoring someone in distress can be hurtful and unkind.'
      ],
      [
        "Borrow a stranger's phone: Unwise, as it's a security risk and may lead to theft or damage.", 
        "Use public charging station: Safe and convenient, as it's a designated charging area.", 
        'Leave your phone unattended while it charges: Not recommended, as it may be stolen or damaged.'
      ],
      [
        'Rush through traffic: This option is risky and may lead to accidents or stress.', 
        'Call and inform them you will be late: This is the most likely option, as it shows respect for the meeting and allows for adjustments.', 
        'Do not show up at all: This is unacceptable, as it shows disrespect for the meeting and may damage relationships.'
      ],
    ],
    answers=[
      "Call for help.",
      "Listen to them and offer comfort.",
      "Use public charging station.",
      "Call and inform them you will be late."
    ],
  )
```

### Models

The retriever model can be found on 🤗 Hugging Face.

- 🦓 **Zebra Retriever**: [`sapienzanlp/zebra-retriever-e5-base-v2`](https://huggingface.co/sapienzanlp/zebra-retriever-e5-base-v2)

### Data

ZEBRA comes with a knowledge base called ZEBRA-KB containing examples of questions along with their automatically-generated list of explanations. \
This KB is where the retriever fetches relevant examples for the input question during the knowledge generation step. \
The KB is organized in two components: the explanations and the document indexes.

The explanations are organized in splits, one for each dataset. Each sample contains an id (compliant with the orginial sample id in the relative dataset) and a list of explanations. There is also a dedicated split which contains the samples of every split.
- **ZEBRA-KB Explanations** [`sapienzanlp/zebra-kb-explanations`](https://huggingface.co/datasets/sapienzanlp/zebra-kb-explanations)

Alternatively, you can also download the expalantions on your local machine from the following [Google Drive link](https://drive.google.com/file/d/1j4SDcaZRdazdpqw4Ei281kGkRYe36sgd/view?usp=drive_link). \
For convenience, we provide a dedicated folder to store the downloaded explanations: `data/explanations`.

The document indexes contain the examples along with their embeddings. These indexes are needed to fetch relevant examples for a given input through the retriever. Once the examples are retrieved, their ids will be matched against the ones contained in the relative explanations split to create the desidered input for the knowledge generation step, that is, a list of k examples with their associated explanations.

- **ZEBRA-KB Document Indexes** [`sapienzanlp/zebra-kb`](https://huggingface.co/sapienzanlp/zebra-kb)

### Reproducibility

We provide a script to run the entire ZEBRA pipeline offline over a dataset using a specific LLM. \
You can find the available datasets for evaluation under the `data/datasets` folder. \
The script expects only one input parameter: the model to be evaluated using the relative HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).

Example on CSQA:

```bash
bash scripts/evaluation/csqa-dev.sh meta-llama/Meta-Llama-3-8B-Instruct
```

We also provide a script to run ZEBRA over all the datasets.

```bash
bash scripts/evaluation/zebra.sh meta-llama/Meta-Llama-3-8B-Instruct
```

These scripts will call **zebra/run_zebra.py** with a predefined set of parameters.
If you wish to run additional experiments by modifying these parameters, you can either modify them directly inside the bash scripts or use the ZEBRA CLI.

### 🦓 ZEBRA CLI

ZEBRA provides a CLI to perform inference on a dataset file. The CLI can be used as follows:

```bash
python zebra/run_zebra.py --help

  Usage: run_zebra.py [ARGUMENTS] [OPTIONS] 

╭─ Arguments ────────────────────────────────────────────────────────────╮
│ *    model_name              TEXT  [default: None] [required]          │
│ *    retriever_output_path   TEXT  [default: None] [required]          │
│ *    data_path               TEXT  [default: None] [required]          │
│ *    explanations_path       TEXT  [default: None] [required]          │
│ *    dataset_tag             TEXT  [default: None] [required]          │
│ *    output_dir              TEXT  [default: None] [required]          │
╰────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────╮
│ --fewshot_data_path                        TEXT     [default: None]    │
│ --explanations_split                       TEXT     [default: None]    │
│ --plain                                    BOOL     [default: False]   │
│ --oracle                                   BOOL     [default: False]   │
│ --examples                                 BOOL     [default: False]   │
│ --max_generated_knowledge                  INTEGER  [default: None]    │
│ --add_negative_explanations                BOOL     [default: False]   │
│ --num_kg_examples                          INTEGER  [default: 1]       │
│ --num_qa_examples                          INTEGER  [default: 0]       │
│ --limit_samples                            INTEGER  [default: None]    │
│ --device                                   TEXT     [default: 'cuda']  │
│ --help                                     Show this message and exit. │
╰────────────────────────────────────────────────────────────────────────╯
```

For example:

```bash
python zebra/run_zebra.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/datasets/csqa/csqa-dev.jsonl \
  --retriever_output_path data/zebra_retriever/outputs/csqa/csqa_dev.csqa_train.jsonl \
  --fewshot_data_path data/datasets/csqa/csqa-train.jsonl \
  --explanations_path sapienzanlp/zebra-kb-explanations \
  --explanations_split csqa-train-gemini
  --dataset_tag csqa \ 
  --output_dir results/zebra/csqa \
  --plain \
  --oracle \
  --examples \
  --num_qa_examples 0 \
  --num_kg_examples 5
```

## 📚 Before You Start

In the following sections, we provide a step-by-step guide on how to prepare the data to test ZEBRA on your dataset, train the ZEBRA retriever and evaluate the models.

### Data Evaluation Format

To be able to run ZEBRA on your dataset, the data should have the following structure:

```jsonl
{
  "id": str,  # Unique identifier for the question
  "question": dict  # Dictionary of the question
    {
      "stem": str, # The question text
      "choices": list[dict] # The object containing the choices for the question
        [
          {
            "label": str, # A Label for every choice, like "A", "B", "C" etc.
            "text": str # The choice text
          },
          ...
        ]
    }
  "answerKey": str # The correct label among the choices
}
```

All the datasets in the `data/datasets` folder already match this format. \
For convenience, we provide a script to parse a dataset in the desired format: `zebra/data/parse_dataset.py`.

### Retriever Training

We trained our retriever on the CSQA dataset [(Talmor et. al 2019)](https://aclanthology.org/N19-1421/). In particular, we format both the training and validation datasets as follows: 

```jsonl
{
  "question": str  # The input passage
  "positive_ctxs": list[dict] # List of positive passages
    [
      {
        "text": str # The text of the positive passage
      },
      ...
    ]
}
```

Where each *question* and *positive* passage is formatted as:

```string
Q [SEP] C1 [SEP] C2 ... [SEP] Cn 
```

To train the retriever with this dataset, you can run

```bash
bash scripts/zebra_retriever/train.sh
```

The script will call **zebra/retriever/train.py** with a predefined set of parameters.
The data needed to train the retriever can be found in the `data/zebra_retriever/datasets` folder.
If you wish to run additional experiments by modifying these parameters, you can either modify them directly inside the bash scripts or use the ZEBRA Retriever CLI.

You can run the following command for more details about training parameters.

```bash
python zebra/retriever/train.py --help

  Usage: train.py [ARGUMENTS] [OPTIONS] 

╭─ Arguments ───────────────────────────────────────────────────────────────────────────────╮
│ *    train_data_path         TEXT  [default: None] [required]                             │
│ *    dev_data_path           TEXT  [default: None] [required]                             │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────╮
│ --question_encoder                         TEXT     [default: intfloat/e5-base-v2]        │
│ --passage_encoder                          TEXT     [default: None]                       │
│ --document_index                           INTEGER  [default: None]                       │
│ --device                                   TEXT     [default: cuda]                       │
│ --precision                                TEXT     [default: 16]                         │
│ --question_batch_size                      INTEGER  [default: 64]                         │
│ --passage_batch_size                       INTEGER  [default: 200]                        │
│ --max_question_length                      INTEGER  [default: 256]                        │
│ --max_passage_length                       INTEGER  [default: 256]                        │
│ --max_steps                                INTEGER  [default: 25000]                      │
│ --num_workers                              INTEGER  [default: 4]                          │
│ --max_hard_negatives_to_mine               INTEGER  [default: 0]                          │
│ --wandb_online_mode                        BOOL     [default: False]                      │
│ --wandb_log_model                          BOOL     [default: False]                      │
│ --wandb_project_name                       TEXT     [default: zebra-retriever]            │
│ --wandb_experiment_name                    TEXT     [default: zebra-retriever-e5-base-v2] │
│ --help                                     Show this message and exit.                    │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

For example:

```bash
python zebra/retriever/train.py \
  --train_data_path data/zebra_retriever/datasets/train.jsonl \
  --dev_data_path data/zebra_retriever/datasets/dev.jsonl
```


### 🦓 ZEBRA KB

As previously explained under the [Data](#data) section, the ZEBRA pipeline requires a document index containing examples such that the retriever can fetch the most relevant ones for an input question. \
The document index can contain:
- examples from the training split of the dataset under evaluation.
- examples from the training split of another dataset.
- examples from multiple training splits of a list of datasets.

You can either access the precomputed document indexes using the link provided in the [Data](#data) section, or you can generate your own document index by running:

```bash
python zebra/retriever/create_index.py --help

  Usage: create_index.py [ARGUMENTS] [OPTIONS] 

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────╮
│ *    data_paths              TEXT  [default: None] [required]                           │
│ *    output_dir              TEXT  [default: None] [required]                           │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────╮
│ --retriever_path             TEXT     [default: sapienzanlp/zebra-retriever-e5-base-v2] │
│ --batch_size                 INTEGER  [default: 512]                                    │
│ --num_workers                INTEGER  [default: 4]                                      │
│ --max_length                 INTEGER  [default: 512]                                    │
│ --device                     TEXT     [default: cuda]                                   │
│ --precision                  TEXT     [default: fp32]                                   │
│ --help                       Show this message and exit.                                │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
```

For example: 

```bash
python zebra/retriever/create_index.py \
  --retriever_path sapienzanlp/zebra-retriever-e5-base-v2 \
  --data_path data/datasets/csqa/csqa-train.jsonl \
  --output_dir data/zebra_retriever/zebra_kb/csqa_train \
```

For convenience, we provide a folder to store the document indexes: `data/zebra_retriever/zebra_kb`.

### Retriever Inference

Once you have a retriever and a document index of the ZEBRA KB, you can retrieve the most relevant examples from the KB for a given input question by running:

```bash
python zebra/retriever/retriever_inference.py --help

  Usage: retriever_inference.py [ARGUMENTS] [OPTIONS] 

╭─ Arguments ─────────────────────────────────────────────────────────────────────────────╮
│ *    data_path               TEXT  [default: None] [required]                           │
│ *    output_path             TEXT  [default: None] [required]                           │
│ *    document_index_path     TEXT  [default: None] [required]                           │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────╮
│ --retriever_path             TEXT     [default: sapienzanlp/zebra-retriever-e5-base-v2] │
│ --batch_size                 INTEGER  [default: 64]                                     │
│ --k                          INTEGER  [default: 100]                                    │
│ --device                     TEXT     [default: cuda]                                   │
│ --help                       Show this message and exit.                                │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
```

For example:

```bash
python zebra/retriever/retriever_inference.py \
  --retriever_path sapienzanlp/zebra-retriever-e5-base-v2 \
  --data_path data/datasets/csqa/csqa-dev.jsonl \
  --document_index_path sapienzanlp/zebra-kb-csqa-train\
  --output_path data/zebra_retriever/outputs/csqa/csqa_dev.csqa_train.jsonl \
```

For convenience, we provide a folder to store the retriever outputs: `data/zebra_retriever/outputs`.

Once you have obtained the retriever's output for a given dataset, you can run the code explained under the [ZEBRA CLI](#-zebra-cli) section to obtain the output and the scores of the ZEBRA pipeline on that dataset.

If you wish to reproduce our results, we also provide the output of our [`retriever`](https://huggingface.co/sapienzanlp/zebra-retriever-e5-base-v2) for all the datasets at the following [Google Drive link](https://drive.google.com/file/d/1HFk_1pnIBN-3bDGm5Bx7d34mPpDjVHRz/view?usp=drive_link).

## 📊 Performance

We evaluate the performance of ZEBRA on 8 well-established commonsense question answering datasets. The following table shows the results (accuracy) of the models before and after the application of ZEBRA.

|          Model           |       CSQA      |      ARC-C      |      ARC-E      |       OBQA      |       PIQA      |       QASC      |      CSQA2      |        WG       |       AVG       |  
| ------------------------ | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | 
| Mistral-7B-Instruct-v0.2 | 68.2 / **73.3** | 72.4	/ **75.2** | 85.8	/ **87.4** | 68.8	/ **75.8** | 76.1	/ **80.2** | 66.1	/ **68.3** | 58.5	/ **67.5** | 55.8 / **60.7** | 68.9 / **73.5** |
| Phi3-small-8k-Instruct   | 77.2 / **80.9** | 90.4 / **91.6** | 96.9	/ **97.7** | 90.4	/ **91.2** | 86.6	/ **88.1** | **83.5**	/ 81.0 | 68.0	/ **74.6** | 79.1	/ **81.0** | 84.0 / **85.8** | 
| Meta-Llama-3-8b-Instruct | 73.9 / **78.7** | 79.4 / **83.5** | 91.7	/ **92.9** | 73.4	/ **79.6** | 78.3	/ **84.0** | 78.2	/ **79.1** | 64.3	/ **69.4** | 56.2	/ **63.2** | 74.4 / **78.8** | 
| Phi3-mini-128k-Instruct  | 73.4 / **74.8** | 85.7	/ **88.0** | 95.4	/ **96.0** | 82.8	/ **87.8** | 80.4	/ **84.2** | **74.7**	/ 73.9 | 59.3	/ **64.6** | 67.3	/ **72.9** | 77.4 / **80.5** | 

You can also download the official paper results at the following [Google Drive Link](https://drive.google.com/file/d/1l7bY-TkqnmVQn5M5ynQfT-0upMcRlMnT/view?usp=drive_link).

## [Baselines](baselines/README.md)

## Cite this work

If you use any part of this work, please consider citing the paper as follows:

[TODO]

## 🪪 License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).