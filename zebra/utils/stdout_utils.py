import csv
import os
from typing import Dict, List, Tuple, Optional, Any


def get_output_path(
    data_path: str,
    model_name: str,
    output_dir: str,
    retriever_output_path: str,
    explanations_path: str,
    explanations_split: Optional[str]=None,
    num_kg_examples: Optional[int]=5,
    num_qa_examples: Optional[int]=0,
    add_negative_explanations: Optional[bool]=False,
) -> Tuple:
    """
    Get the output path for the results.

    Parameters:
    - data_path (str): Path to the dataset.
    - model_name (str): Name of the model.
    - output_dir (str): Directory where the output will be saved.
    - retriever_output_path (str): Path to the retriever output.
    - explanations_path (str): Path to the explanations.
    - explanations_split (Optional[str]): Split of the explanations. Default is None.
    - num_kg_examples (Optional[int]): Number of knowledge graph examples. Default is 5.
    - num_qa_examples (Optional[int]): Number of QA examples. Default is 0.
    - add_negative_explanations (Optional[bool]): Whether to add negative explanations. Default is False.

    Returns:
    - Tuple: A tuple containing the output path and scores path.
    """
    if not os.path.exists(retriever_output_path):
        raise ValueError(f"Retriever output path {retriever_output_path} does not exist.")

    dataset_output_tag = data_path.split("/")[-1].split(".")[0]
    model_tag = model_name.replace("/", "_")
    explanations_tag = None
    if os.path.exists(explanations_path):
        explanations_tag = explanations_path.split("/")[-1].split(".")[0]
    else:
        explanations_tag = explanations_path.split("/")[-1] + "-" + explanations_split
    retriever_output_tag = retriever_output_path.split("/")[-1].split(".")[-2]
    output_filename = f"{dataset_output_tag}|{model_tag}|num_kg_shots={num_kg_examples}|num_qa_shots={num_qa_examples}|add_negative_explanations={add_negative_explanations}|explanations={explanations_tag}|document_index={retriever_output_tag}.tsv"
    scores_filename = f"scores|{dataset_output_tag}|{model_tag}|num_kg_shots={num_kg_examples}|num_qa_shots={num_qa_examples}|add_negative_explanations={add_negative_explanations}|explanations={explanations_tag}|document_index={retriever_output_tag}.tsv"
    output_path = os.path.join(output_dir, output_filename)
    scores_path = os.path.join(output_dir, scores_filename)

    # Create output directory if it does not exist.
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    return output_path, scores_path


def write_header(writer: csv.writer):
    """
    Write the header to the file.

    Parameters:
    - writer (csv.writer): CSV writer.
    """
    writer.writerow(
        [
            "question",
            "examples",
            "examples_knowledge",
            "example_choices",
            "example_answers",
            "oracle_knowledge",
            "generated_knowledge",
            "choices",
            "ground_truth",
            "answer_without_knowledge",
            "answer_with_knowledge",
            "answer_with_oracle_knowledge",
        ]
    )


def write_row(
    writer: csv.writer,
    question: Optional[str]=None,
    retrieved_examples: Optional[List[Dict[str, Any]]]=[],
    choices: Optional[List[Dict[str, str]]]=[],
    ground_truth: Optional[str]=None,
    oracle_knowledge: Optional[List[str]]=[],
    generated_knowledge: Optional[List[str]]=[],
    answer_without_knowledge: Optional[str]=None,
    answer_with_knowledge: Optional[str]=None,
    answer_with_oracle_knowledge: Optional[str]=None,
    num_kg_examples: Optional[int]=5,
):
    """
    Write a row to the file.

    Parameters:
        writer (csv.writer): CSV writer.
        question (Optional[str]): Question.
        retrieved_examples (Optional[List[Dict[str, Any]]]): Retrieved examples.
        choices (Optional[List[Dict[str, str]]]): Choices.
        ground_truth (Optional[str]): Ground truth.
        oracle_knowledge (Optional[List[str]]): Oracle knowledge.
        generated_knowledge (Optional[List[str]]): Generated knowledge.
        answer_without_knowledge (Optional[str]): Answer without knowledge.
        answer_with_knowledge (Optional[str]): Answer with knowledge.
        answer_with_oracle_knowledge (Optional[str]): Answer with oracle knowledge.
        num_kg_examples (Optional[int]): Number of knowledge graph examples.
    """
    writer.writerow(
        [
            question,
            "\n\n".join(
                [re["question"] for re in retrieved_examples[:num_kg_examples]]
            ),
            "\n\n".join(
                [
                    "\n".join(re["knowledge"])
                    for re in retrieved_examples[:num_kg_examples]
                ]
            ),
            "\n\n".join(
                [
                    "\n".join(
                        [
                            f"{choice['label']}. {choice['text']}"
                            for choice in re["choices"]
                        ]
                    )
                    for re in retrieved_examples[:num_kg_examples]
                ]
            ),
            "\n\n".join([re["answer"] for re in retrieved_examples[:num_kg_examples]]),
            "\n".join(oracle_knowledge),
            "\n".join(generated_knowledge),
            "\n".join([f"{choice['label']}. {choice['text']}" for choice in choices]),
            ground_truth,
            answer_without_knowledge,
            answer_with_knowledge,
            answer_with_oracle_knowledge,
        ]
    )


def write_scores(
    scores_path: str,
    metrics: Dict[str, float],
):
    """
    Write the scores to a file.

    Parameters:
    - scores_path (str): Path to the file.
    - metrics (Dict[str, float]): Metrics.
    """
    accuracy_without_knowledge = metrics["accuracy_without_knowledge"]
    accuracy_with_knowledge = metrics["accuracy_with_knowledge"]
    accuracy_with_oracle_knowledge = metrics["accuracy_with_oracle_knowledge"]

    with open(scores_path, "w") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(
            [
                "accuracy",
                "accuracy_with_knowledge",
                "accuracy_with_oracle_knowledge",
            ]
        )
        writer.writerow(
            [
                accuracy_without_knowledge,
                accuracy_with_knowledge,
                accuracy_with_oracle_knowledge,
            ]
        )
