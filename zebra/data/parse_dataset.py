import argparse
from itertools import repeat

import jsonlines
import tqdm


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add arguments.
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        required=True,
        help="Path to the dataset to be parsed.",
    )
    parser.add_argument(
        "--dataset_tag",
        "-dt",
        type=str,
        required=True,
        help="Dataset tag (e.g. csqa2, piqa, wg).",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=True,
        help="Dataset split (e.g. dev, test, train).",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=str,
        required=True,
        help="Output path of the parsed dataset.",
    )
    # Add optional arguments.
    parser.add_argument(
        "--labels_path",
        "-lp",
        type=str,
        required=False,
        default=None,
        help="Path to the label file (if any).",
    )
    # Parse arguments.
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments.
    args = parse_arguments()

    with open(args.data_path, "r") as input_data:
        data = input_data.readlines()

    if args.labels_path is not None:
        with open(args.labels_path, "r") as input_labels:
            labels = input_labels.readlines()

    with jsonlines.open(args.output_path, "w") as writer:
        iterator = tqdm.tqdm(
            enumerate(zip(data, labels if labels is not None else repeat(None))),
            "Generating answers...",
        )
        for i, (line, label) in iterator:
            new_line = {}
            if "id" not in line and "qID" not in line:
                new_line["id"] = args.split + "-" + str(i)
            elif "id" in line and "qID" in line:
                new_line["id"] = line["qID"]
            else:
                new_line["id"] = line["id"]

            if "csqa2" in args.dataset_tag:
                new_line["answerKey"] = "A" if line["answer"] == "yes" else "B"
                new_line["question"] = {
                    "stem": line["question"],
                    "choices": [
                        {"label": "A", "text": "Yes"},
                        {"label": "B", "text": "No"},
                    ],
                }
            elif "piqa" in args.dataset_tag and label is not None:
                new_line["answerKey"] = "A" if label.strip() == "0" else "B"
                new_line["question"] = {
                    "stem": line["goal"],
                    "choices": [
                        {"label": "A", "text": line["sol1"]},
                        {"label": "B", "text": line["sol2"]},
                    ],
                }

            elif "wg" in args.dataset_tag:
                new_line["answerKey"] = "A" if line["answer"] == "1" else "B"
                new_line["question"] = {
                    "stem": line["sentence"],
                    "choices": [
                        {"label": "A", "text": line["option1"]},
                        {"label": "B", "text": line["option2"]},
                    ],
                }

            writer.write(new_line)
