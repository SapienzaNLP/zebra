from argparse import ArgumentParser
from collections import defaultdict

import os
import json
import jsonlines
import torch
from goldenretriever import GoldenRetriever
from tqdm import tqdm

RETRIEVER_PATH = "sapienzanlp/zebra-retriever-e5-base-v2"


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Command-line arguments.
    """
    parser = ArgumentParser()

    # Add arguments.
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the retrieved data.",
    )
    parser.add_argument(
        "--document_index_path",
        type=str,
        required=True,
        help="Path to the document index. Can be either a local path or a dataset ID from Hugging Face.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--retriever_path",
        type=str,
        default=RETRIEVER_PATH,
        help="Path to the retriever model. Can be either a local path or a model ID from Hugging Face.",
    )
    parser.add_argument(
        "--k", type=int, default=100, help="Number of examples to retrieve."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for retrieval."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default is 'cuda').",
    )

    # Parse arguments.
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The data path {args.data_path} does not exist. Please provide a valid path.")
    
    with open(args.data_path, "r") as fin:
        dataset = [json.loads(line) for line in fin]

    passages = []
    passage_ids = []
    passage_to_id = defaultdict(str)

    for sample in dataset:
        question = sample["question"]["stem"]
        choices = sample["question"]["choices"]
        joined_choices = " [SEP] ".join(
            [f"{choice['label']}. {choice['text']}" for choice in choices]
        )
        passage = f"{question} [SEP] {joined_choices}"
        passages.append(passage)
        passage_ids.append(sample["id"])
        passage_to_id[passage] = sample["id"]

    ## Instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=args.retriever_path,
        document_index=args.document_index_path,
        device=args.device,
    )

    retrieved_examples = retriever.retrieve(
        passages, 
        k=args.k, 
        batch_size=args.batch_size
    )

    with jsonlines.open(args.output_path, "w") as fout:
        for passage, example in tqdm(
            zip(passages, retrieved_examples), "Writing Result..."
        ):
            query = {"text": passage, "qid": passage_to_id[passage]}
            candidates = [
                {
                    "docid": e.document.id,
                    "score": e.score,
                    "doc": {
                        "id": e.document.id,
                        "contents": e.document.text,
                        "metadata": {
                            "answerKey": e.document.metadata["answerKey"],
                        },
                    },
                }
                for e in example
            ]
            request = {"query": query, "candidates": candidates}
            fout.write(request)
