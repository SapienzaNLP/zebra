import argparse
import json
import os
from collections import defaultdict

import jsonlines
import torch
from goldenretriever import GoldenRetriever
from tqdm import tqdm

RETRIEVER_PATH = "sapienzanlp/rbr-retriever-gkb-omcs-atomic"


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add arguments.
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the predictions.",
    )
    parser.add_argument(
        "--document_index_path",
        type=str,
        required=True,
        help="Path to the document index.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--retriever_path",
        type=str,
        default=RETRIEVER_PATH,
        help="Path to the retriever model.",
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default is 'cuda').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.data_path, "r") as fin:
        dataset = fin.readlines()

    questions = []
    questions_ids = []
    question_to_id = defaultdict(str)

    for sample in dataset:
        sample = json.loads(sample)
        questions.append(sample["question"]["stem"])

    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=RETRIEVER_PATH, document_index=args.document_index_path
    ).to(args.device)

    # Retrieve knowledge
    retrieved_knowledge = retriever.retrieve(
        questions, k=args.k, batch_size=args.batch_size
    )

    # Get output path.
    dataset_tag = args.data_path.split("/")[-1].split(".")[0]
    output_filename = f'{dataset_tag}|{args.retriever_path.split("/")[1]}|explanations=rbr|num_return_sequences={args.k}.jsonl'
    output_path = os.path.join(args.output_dir, output_filename)

    # Create output directory if it does not exist.
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Write results to output file.
    with jsonlines.open(output_path, "w") as fout:
        for sample, list_of_knowledge in tqdm(
            zip(dataset, retrieved_knowledge), "Writing Result..."
        ):
            sample = json.loads(sample)
            knowledge = [k.document.text for k in list_of_knowledge]
            sample["knowledge"] = knowledge
            fout.write(sample)
