
<div align="center">
  <img src="https://github.com/SapienzaNLP/zebra/blob/master/assets/zebra.png?raw=true" width="100" height="100">
</div>

<div align="center">

# ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering

[![Conference](https://img.shields.io/badge/EMNLP-2024-4b44ce)](https://2024.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2410.05077)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/sapienzanlp/zebra-66e3ec50c8ce415ea7572d0e)
[![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
</div>

<div align="center"> A retrieval augmentation framework for zero-shot commonsense question answering with LLMs. </div>

## 🛠️ Installation

Installation from PyPi

```bash
pip install zebra-qa
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
It is composed of three pipeline stages: *example retrieval*, *knowledge generation* and *informed reasoning*.

- Example retrieval: given a question, we retrieve relevant examples of question-knowledge pairs from a large collection
- Knowledge generation: we prompt an LLM to generate useful explanations for the given input question by leveraging the relationships in the retrieved question-knowledge pairs.
- Informed reasoning: we prompt the same LLM for the question answering task by taking advantage of the previously generated explanations.

Here is an example of how to use ZEBRA for question answering:

```python
from zebra import Zebra

# Load Zebra with language model, retriever, document index and explanations.
zebra = Zebra(
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  retriever="sapienzanlp/zebra-retriever-e5-base-v2",
  document_index="sapienzanlp/zebra-kb"
)

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

The output contains, for each question, a list of generated explanations and the predicted answer:

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

You can also call the `zebra.pipeline` method with the `return_dict` parameter set to `True` to ask ZEBRA to return also the retrieved examples along with their explanations.

## Retriever Model

We trained our retriever on the CSQA dataset [(Talmor et. al 2019)](https://aclanthology.org/N19-1421/).
The retriever model can be found on 🤗 Hugging Face.

- 🦓 **Zebra Retriever**: [`sapienzanlp/zebra-retriever-e5-base-v2`](https://huggingface.co/sapienzanlp/zebra-retriever-e5-base-v2)

## Data

ZEBRA comes with a knowledge base called ZEBRA-KB containing examples of questions along with their automatically-generated list of explanations. To create the explanations, we prompt Google Gemini-1.5-Flash to generate useful knowledge given a question together with its choices and correct answer. The examples are taken from the training sets of the following question answering benchmarks:

| Dataset | Description | Link |
|---------|-------------|------|
| CSQA    | CommonsenseQA is a dataset for commonsense question answering. | [CSQA](https://www.tau-nlp.org/commonsenseqa) |
| ARC     | AI2 Reasoning Challenge is a dataset for science question answering. | [ARC](https://allenai.org/data/arc) |
| OBQA    | OpenBookQA is a dataset for open book question answering. | [OBQA](https://allenai.org/data/open-book-qa) |
| PIQA    | Physical Interaction QA is a dataset for physical commonsense reasoning. | [PIQA](https://yonatanbisk.com/piqa/) |
| QASC    | Question Answering via Sentence Composition is a dataset for multi-hop question answering. | [QASC](https://allenai.org/data/qasc) |
| CSQA2   | CommonsenseQA 2.0 is a dataset for commonsense question answering. | [CSQA2](https://github.com/allenai/csqa2) |
| WG      | Winograd Schema Challenge is a dataset for commonsense reasoning. | [WG](https://github.com/allenai/winogrande) |


This KB is where the retriever fetches relevant examples for the input question. The KB is organized in two components: the explanations and the document indexes.

The explanations are organized in splits, one for each training set (e.g. `csqa-train-gemini`). Each sample contains an ID (compliant with the original sample ID in the relative training set) and a list of explanations. There is also a dedicated split which contains all the samples of every split. You can access the explanations at the following link:

- **ZEBRA-KB Explanations** [`sapienzanlp/zebra-kb-explanations`](https://huggingface.co/datasets/sapienzanlp/zebra-kb-explanations)

Alternatively, you can also download the explanations on your local machine from the following [Google Drive link](https://drive.google.com/file/d/1eKBB1DaQQx-s5ibiZrrgfZpfDwDMxxWB/view?usp=sharing). For convenience, we provide a dedicated folder to store the downloaded explanations: `data/explanations`.

The document indexes contain the examples along with their embeddings. These indexes are needed to fetch relevant examples for a given input question through the retriever. Once the examples are retrieved, their IDs will be matched against the ones contained in the relative explanations split to create the desired input for the knowledge generation step, that is, a list of $k$ examples with their associated explanations.

Similar to the explanations, the document indexes are organized in splits, one for each training set. You can browse the available splits at the following [HuggingFace Collection link](https://huggingface.co/collections/sapienzanlp/zebra-66e3ec50c8ce415ea7572d0e).

We also provide a document index containing the splits of every training set:

- **ZEBRA-KB Document Index** [`sapienzanlp/zebra-kb`](https://huggingface.co/sapienzanlp/zebra-kb)

## Reproducibility

If you wish to reproduce our results, we provide the output of our [`retriever`](https://huggingface.co/sapienzanlp/zebra-retriever-e5-base-v2) for all the datasets at the following [Google Drive link](https://drive.google.com/file/d/1HFk_1pnIBN-3bDGm5Bx7d34mPpDjVHRz/view?usp=drive_link).

After you have downloaded the zip file, please unzip it and move its contents to the `data/retriever/outputs` folder. Then, you should be able to see something like `data/retriever/outputs/{dataset}` with some .jsonl files inside. Each .jsonl file contains the top $k=100$ examples fetched by the retriever for each input question of the dataset. The naming convention of the .jsonl file is: `{dataset}_{split}.{dataset}_train.jsonl`, where `{dataset}_{split}` specifies the dataset from which the input questions are drawn from (e.g. `csqa_dev`), while `{dataset}_train` specifies the document index in ZEBRA-KB from which the examples are drawn from (e.g. `csqa_train`).

We provide a script to run the entire ZEBRA pipeline offline over a dataset using a specific LLM. You can find the available datasets files for evaluation under the `data/datasets` folder. Once you have placed the retriever's outputs in the dedicated folder, the script expects only one input parameter: the model to be evaluated using the relative HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).

Example on CSQA:

```bash
bash scripts/evaluation/csqa-dev.sh meta-llama/Meta-Llama-3-8B-Instruct
```

We also provide a script to run ZEBRA over all the datasets.

```bash
bash scripts/evaluation/zebra.sh meta-llama/Meta-Llama-3-8B-Instruct
```

These scripts will call **scripts/evaluation/run_zebra.py** with a predefined set of parameters.
If you wish to run additional experiments by modifying these parameters, you can either modify them directly inside the bash scripts or call the python script with command line arguments.

```bash
python scripts/evaluation/run_zebra.py --help

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
python scripts/evaluation/run_zebra.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/datasets/csqa/csqa-dev.jsonl \
  --retriever_output_path data/retriever/outputs/csqa/csqa_dev.csqa_train.jsonl \
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

## 🦓 Zebra Pipeline

In the following sections, we provide a step-by-step guide on how to prepare the data to test ZEBRA on your dataset, train your own ZEBRA retriever and evaluate the models.

### Data Evaluation Format

To be able to run ZEBRA on your dataset, the data should have the following structure:

```python
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

All the datasets in the `data/datasets` folder already match this format. For convenience, we provide a script to parse a dataset in the desired format: `scripts/data/parse_dataset.py`.

### Retriever Training

Our retriever model can be found at the link in the [Models](#models) section.
We trained our retriever on the CSQA dataset [(Talmor et. al 2019)](https://aclanthology.org/N19-1421/). In particular, we format both the training and validation datasets as follows: 

```python
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

If you wish to train your own retriever with this dataset, you can run:

```bash
bash scripts/retriever/train.sh
```

The script will call **scripts/retriever/train.py** with a predefined set of parameters.
The data needed to train the retriever can be found in the `data/retriever/datasets` folder.
If you wish to run additional experiments by modifying these parameters, you can either modify them directly inside the bash scripts or call the python script with command line arguments.

```bash
python scripts/retriever/train.py --help

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
python scripts/retriever/train.py \
  --train_data_path data/retriever/datasets/train.jsonl \
  --dev_data_path data/retriever/datasets/dev.jsonl
```


### 🦓 ZEBRA KB

As previously explained under the [Data](#data) section, the ZEBRA pipeline requires a document index containing examples such that the retriever can fetch the most relevant ones for an input question. The document index can contain:
- examples from the training set of the dataset under evaluation.
- examples from the training set of another dataset.
- examples from multiple training sets of a list of datasets.

You can either access the precomputed document indexes using the link provided in the [Data](#data) section, or you can generate your own document index by running:

```bash
python scripts/retriever/create_index.py --help

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
  --output_dir data/retriever/zebra_kb/csqa_train \
```

For convenience, we provide a folder to store the document indexes: `data/retriever/zebra_kb`.

### Retriever Inference

Once you have a retriever and a document index of the ZEBRA KB, you can retrieve the most relevant examples from the KB for a given input question by running:

```bash
python scripts/retriever/retriever_inference.py --help

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
  --output_path data/retriever/outputs/csqa/csqa_dev.csqa_train.jsonl \
```

For convenience, we provide a folder to store the retriever outputs: `data/retriever/outputs`.

Once you have obtained the retriever's output for a given dataset, you can run the code explained under the [Reproducibility](#Reproducibility) section to obtain the output and the scores of the ZEBRA pipeline on that dataset.

## 📊 Performance

We evaluate the performance of ZEBRA on 8 well-established commonsense question answering datasets. The following table shows the results (accuracy) of the models before / after the application of ZEBRA.

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

```bibtex
@inproceedings{molfese-etal-2024-zebra,
    title = "ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering",
    author = "Molfese, Francesco Maria  and
      Conia, Simone  and
      Orlando, Riccardo  and
      Navigli, Roberto",
    editor = "",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
    abstract = "",
}
```

## 🪪 License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements
We gratefully acknowledge CREATIVE (CRoss-modalunderstanding and gEnerATIon of Visual and tExtual content) for supporting this work. Simone Conia gratefully acknowledges the support of Future AI Research ([PNRR MUR project PE0000013-FAIR](https://fondazione-fair.it/en/)), which fully funds his fellowship at Sapienza University of Rome since October 2023.
