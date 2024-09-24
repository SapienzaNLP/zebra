import csv
import os
from typing import Dict, List, Tuple


def get_output_path(
    data_path: str,
    model_name: str,
    baseline: str,
    max_knowledge: int,
    output_dir: str,
) -> Tuple:
    """
    Get the output path for the results.
    """
    dataset_output_tag = data_path.split("/")[-1].split("|")[0]
    model_tag = model_name.replace("/", "_")
    output_filename = f"{dataset_output_tag}|{model_tag}|explanations={baseline}|max_knowledge={max_knowledge}.tsv"
    scores_filename = f"scores|{dataset_output_tag}|{model_tag}|explanations={baseline}|max_knowledge={max_knowledge}.tsv"
    output_path = os.path.join(output_dir, output_filename)
    scores_path = os.path.join(output_dir, scores_filename)

    # Create output directory if it does not exist.
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    return output_path, scores_path


def write_header(writer: csv.writer):
    """
    Write the header to the file.

    Args:
        writer (csv.writer): CSV writer.
    """
    writer.writerow(
        [
            "question",
            "choices",
            "knowledge",
            "ground_truth",
            "answer_without_knowledge",
            "answer_with_knowledge",
        ]
    )


def write_row(
    writer: csv.writer,
    question: str,
    choices: List[Dict],
    ground_truth: str,
    knowledge: List[str],
    answer_without_knowledge: str,
    answer_with_knowledge: str,
):
    """
    Write a row to the file.

    Args:
        writer (csv.writer): CSV writer.
        question (str): Question.
        choices (List[Dict]): Choices.
        ground_truth (str): Ground truth.
        knowledge (List[str]): Generated knowledge.
        answer_without_knowledge (str): Answer without knowledge.
        answer_with_knowledge (str): Answer with knowledge.
    """
    writer.writerow(
        [
            question,
            "\n".join([f"{choice['label']}. {choice['text']}" for choice in choices]),
            "\n".join(knowledge),
            ground_truth,
            answer_without_knowledge,
            answer_with_knowledge,
        ]
    )


def write_scores(
    scores_path: str,
    metrics: Dict[str, float],
):
    """
    Write the scores to a file.

    :
        scores_path (str): Path to the file.
        accuracy_without_knowledge (float): Accuracy without knowledge.
        accuracy_with_knowledge (float): Accuracy with knowledge.
    """
    accuracy_without_knowledge = metrics["accuracy_without_knowledge"]
    accuracy_with_knowledge = metrics["accuracy_with_knowledge"]

    with open(scores_path, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(
            [
                "accuracy",
                "accuracy_with_knowledge",
            ]
        )
        writer.writerow(
            [
                accuracy_without_knowledge,
                accuracy_with_knowledge,
            ]
        )
