## Baselines

Under the `baselines` folder we provide all the necessary utils to reproduce the baselines' results of our paper.

### Unsupervised methods

Both [Self-Talk](https://aclanthology.org/2020.emnlp-main.373/) and [GKP](https://aclanthology.org/2022.acl-long.225.pdf) provide repository and code to reproduce their experiments.

- [Self-Talk Repository](https://github.com/vered1986/self_talk)
- [GKP Repository](https://github.com/liujch1998/GKP)

For Self-Talk, you can find the necessary code to adapt their framework to support LLMs at the following [Google Drive link](https://drive.google.com/drive/folders/1ght1DoTEftTDABOw5P9FXj1r58JU-2nw?usp=drive_link). \
In particular, you should follow the following steps:
- add the desired datasets from `zebra/data/datasets` to the `self_talk/data` folder (e.g. `cp -r zebra/data/datasets/obqa self_talk/data`)
- rename every dataset with the format: `{split}.jsonl` (e.g. `mv self_talk/data/obqa/obqa-test.jsonl self_talk/data/obqa/test.jsonl`)
- create a folder under `self_talk/experiments` with the same name as the folder containing the dataset (e.g. `mkdir self_talk/experiments/obqa`)
- create a json file called `prefixes.json` where the question clarification prefixes will be stored (e.g `touch self_talk/experiments/obqa/prefixes.json`)
- fill-in the `prefixes.json` file with the desired question perfixes
- replace the `generate_clarifications_from_lm.py` with the one contained in the drive link
- replace the `lm_text_generator.py` with the one contained in the drive link
- add the `generate_llms_clarifications.sh` to the folder to run the pipeline with the newly added scripts

For GKP, you can find the necessary code to adapt their framework to support LLMs at the following [Google Drive link](https://drive.google.com/drive/folders/12uzSxzPhd0SEBWNUEmOWu29NKSCLqExa?usp=drive_link). \
In particular, you should follow the following steps:
- add the desired datasets from `zebra/data/datasets` to the `GKP/data` folder (e.g. `cp -r zebra/data/datasets/obqa GKP/data`)
- add the `{dataset_name}_standardize.py` script contained in the drive link to the `GKP/standardize` folder
- standardize the desired dataset by running the `GKP/standardize/{dataset_name}_standardize.py` script
- add the prompts contained in the drive link to the `GKP/knowledge/prompts` folder
- add the `llms_generate_knowledge.py` script from the drive link to the `GKP/knowledge` folder
- run the `llms_generate_knowledge.py` python script with the required parameters.

These procedures will produce the relative baseline outputs needed by ZEBRA to run their evaluation.
For convenience, in our repository we provide a folder to store the results of the baselines: `baselines/outputs`.

### Supervised methods

Both [Rainier](https://arxiv.org/abs/2210.03078) and [Crystal](https://aclanthology.org/2023.emnlp-main.708.pdf) provide the relative HuggingFace code to load their models and to generate knowledge.

- [Rainier HuggingFace page](https://huggingface.co/liujch1998/rainier-large)
- [Crystal HuggingFace page](https://huggingface.co/liujch1998/crystal-11b)

For convenience, we provide the scripts to generate the knowledge with both Raininer and Crystal under the `baselines` folder:

- Raininer: `baselines/generate_raininer_knowledge.py`
- Crystal (both 3b and 11b): `baselines/generate_crystal_knowledge.py`

For convenience, in our repository we provide a folder to store the results of the baselines: `baselines/outputs`.

### RACo-based Retrieval Augmentation (RBR)

In the [Retrival Augmentation for Commonsense Reasoning: A Unified Approach](https://aclanthology.org/2022.emnlp-main.294/) (RACo) paper, the authors provide guidelines to train a retriever to fetch relevant commonsense statements from a large collection.

Both the data to train the retriever and the knowledge base serving as document index can be found in the official [RACo repository](https://github.com/wyu97/RACo).

Steps:
- Gather the data needed to train the RACo retriever and the documents that will serve as document index from the official RACo repository, then you can train your own RACo retriever by running: `baselines/rbr_retriever/train.py`. To ease reproducibility, we provide a checkpoint of a pre-trained RACo retriever at the following [HuggingFace link](https://huggingface.co/sapienzanlp/rbr-retriever-gkb-omcs-atomic).
Alternatively, you can just follow the steps explained in the official RACo repository to train the retriever.
- Once you have a pre-trained RACo retriever, you can create your own document index by running: `baselines/rbr_retriever/create_index.py`
- Once the document index is created, you can generate the outputs for a specific dataset by running: `baselines/rbr_retriever/retriever_inference.py`

For convenience, in our repository we provide a folder to store the results of the baselines: `baselines/outputs`.

### Evaluation

Once you have generated the output of a relative baseline, you can run: `baselines/evaluate_baseline.py`

```bash
python baselines/evaluate_baseline.py --help

  Usage: evaluate_baseline.py [ARGUMENTS] [OPTIONS] 

╭─ Arguments ────────────────────────────────────────────────────────────╮
│ *    model_name              TEXT  [default: None] [required]          │
│ *    data_path               TEXT  [default: None] [required]          │
│ *    dataset_tag             TEXT  [default: None] [required]          │
│ *    output_dir              TEXT  [default: None] [required]          │
│ *    baseline                TEXT  [default: None] [required]          │
╰────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────╮
│ --plain                                    BOOL     [default: False]   │
│ --max_knowledge                            INTEGER  [default: None]    │
│ --limit_samples                            INTEGER  [default: None]    │
│ --device                                   TEXT     [default: 'cuda']  │
│ --help                                     Show this message and exit. │
╰────────────────────────────────────────────────────────────────────────╯
```

For example:

```bash
python scripts/evaluation/run_zebra.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path baselines/outputs/csqa/csqa-dev|crystal-11b|explanations=crystal|num_return_sequences=10.jsonl \
  --dataset_tag csqa \ 
  --output_dir results/crystal/csqa \
  --baseline crystal
```

To ease reproducibility, we provide all the baselines outputs at the following [Google Drive link](https://drive.google.com/file/d/1Wm4LX_K-MaqhTDv4tAlySJPf8GHV6CTo/view?usp=sharing).
You can download the baseline outputs from the link and put them into the `baselines/outputs` folder.