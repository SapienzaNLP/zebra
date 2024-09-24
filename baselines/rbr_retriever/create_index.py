from argparse import ArgumentParser

import torch
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex

RETRIEVER_PATH = "sapienzanlp/rbr-retriever-gkb-omcs-atomic"


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
        help="Path to the file containing the documents to be indexed. Expects a TSV file with columns 'id' and 'text'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the indexed documents.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--retriever_path",
        type=str,
        default=RETRIEVER_PATH,
        help="Path to the retriever model. Can be either a local path or a model ID from Hugging Face.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for encoding the documents.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for encoding the documents.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length of the input sequences.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for encoding the documents. Defaults to 'cuda' if available.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision for encoding the documents. Defaults to 'fp32'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    documents = DocumentStore.from_tsv(args.data_path)

    document_index = InMemoryDocumentIndex(
        documents=documents,
        device=args.device,
        precision=args.precision,
    )

    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder=args.retriever_path,
        document_index=document_index,
    ).to(args.device)

    retriever.index(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        force_reindex=True,
        precision=args.precision,
    )

    retriever.document_index.save_pretrained(output_dir=args.output_dir)
