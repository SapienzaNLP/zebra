import json
import os
import random
from collections import defaultdict
from itertools import repeat
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from datasets import load_dataset


def load_evaluation_data(data_path: str, limit_samples: Optional[int]=None) -> List[Dict[str, Any]]:
    """
    Load the evaluation data from the given path.

    Parameters:
    - data_path (str): The path to the evaluation data file.
    - limit_samples (int): The maximum number of samples to load from the file.

    Returns:
    - List[Dict[str, Any]]: A list of dictionaries containing the evaluation data.

    Raises:
    - FileNotFoundError: If the specified data_path does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The evaluation data path {data_path} does not exist. Please provide a valid path."
        )
    with open(data_path, "r") as f_in:
        eval_data = f_in.readlines()
        if limit_samples is not None:
            eval_data = eval_data[:limit_samples]
    return eval_data


def load_explanations(
    explanations_path: str, 
    explanations_split: Optional[str]=None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Load the explanations from the given path.

    Parameters:
    - explanations_path (str): The path to the explanations file or HuggingFace dataset ID.
    - explanations_split (Optional[str]): The split of the HuggingFace dataset to load if explanations_path is not a local file.

    Returns:
    - Dict[str, Dict[str, List[str]]]: A dictionary mapping question IDs to their corresponding positive and negative explanations.

    Raises:
    - FileNotFoundError: If the explanations_path is not a local file and explanations_split is not provided, or if the explanations_path is neither a local file nor a valid HuggingFace dataset ID.
    """
    if not os.path.exists(explanations_path):
        if explanations_split is None:
            raise FileNotFoundError(
                f"""The explanations path {explanations_path} is not a local file. 
                Please provide an explanations split to try loading the file 
                using {explanations_path} as an HuggingFace dataset ID."""
            )
        try:
            explanations = load_dataset(explanations_path, explanations_split)["train"]
        except:
            raise FileNotFoundError(
                f"The explanations path {explanations_path} is neither a local file nor a valid HuggingFace dataset ID."
            )
    else:
        with open(explanations_path, "r") as f_in:
            explanations = [json.loads(line) for line in f_in]
    # Load positive and negative explanations.
    question_to_explanations = defaultdict(lambda: {"positives": [], "negatives": []})

    for sample in explanations:
        question_id = sample["id"]
        if "positives" in sample and sample["positives"]:
            question_to_explanations[question_id]["positives"] = sample["positives"]
        if "negatives" in sample and sample["negatives"]:
            question_to_explanations[question_id]["negatives"] = sample["negatives"]

    return question_to_explanations


def create_qa_fewshot_examples(
    len_eval_data: int,
    question_to_explanations: Dict[str, Dict[str, List[str]]],
    fewshot_data_path: Optional[str]=None,
    num_qa_examples: Optional[int]=0,
    upsampling_factor: Optional[int]=5,
    add_negative_explanations: Optional[bool]=False,
) -> List[List[Dict[str, Any]]]:
    """
    Create few-shot examples for question answering tasks.

    Parameters:
    - len_eval_data (int): The length of the evaluation data.
    - question_to_explanations (Dict[str, Dict[str, List[str]]]): A dictionary mapping question IDs to their corresponding positive and negative explanations.
    - fewshot_data_path (Optional[str]): The path to the few-shot data file.
    - num_qa_examples (Optional[int]): The number of question-answer examples to generate.
    - upsampling_factor (Optional[int]): The factor by which to upsample the candidates.
    - add_negative_explanations (Optional[bool]): Whether to include negative explanations.

    Returns:
    - List[List[Dict[str, Any]]]: A list of lists containing the few-shot examples for each evaluation data point.

    Raises:
    - FileNotFoundError: If the fewshot_data_path is provided but does not exist.
    - Warning: If the fewshot_data_path is provided but num_qa_examples is set to 0.
    - Warning: If num_qa_examples is set to a value greater than 0 but fewshot_data_path is not provided.
    """
    qa_shots = list(repeat([], len_eval_data))
    if fewshot_data_path and num_qa_examples > 0:
        if not os.path.exists(fewshot_data_path):
            raise FileNotFoundError(
                f"The few-shot data path {fewshot_data_path} does not exist. Please provide a valid path."
            )
        with open(fewshot_data_path, "r") as f_in:
            fewshot_data = f_in.readlines()

    if num_qa_examples == 0 and fewshot_data_path:
        logger.warning(
            "Fewshot data is provided but num_qa_examples is set to 0. Skipping the creation of fewshot examples for the question answering task."
        )
    elif num_qa_examples > 0 and not fewshot_data_path:
        logger.warning(
            "Fewshot data is not provided but num_qa_examples is set to a value greater than 0. Skipping the creation of fewshot examples for the question answering task."
        )

    elif num_qa_examples > 0 and fewshot_data_path:
        qa_shots = []
        for _ in range(len_eval_data):
            sample_shots = []
            # Upsample the candidates so as to get at least max_shots candidates with explanations.
            # The candidates will be cut down to max_shots at the end.
            candidates = random.sample(
                fewshot_data, num_qa_examples * upsampling_factor
            )

            for candidate in candidates:
                candidate = json.loads(candidate)
                example_id = candidate["id"]
                example_question = candidate["question"]["stem"]
                example_choices = candidate["question"]["choices"]
                example_answer = candidate["answerKey"]

                example_knowledge = question_to_explanations[example_id]["positives"]
                if add_negative_explanations:
                    example_knowledge.extend(
                        question_to_explanations[example_id]["negatives"]
                    )

                if not example_knowledge:
                    continue

                sample_shots.append(
                    {
                        "id": example_id,
                        "question": example_question,
                        "choices": example_choices,
                        "knowledge": example_knowledge,
                        "answer": example_answer,
                    }
                )

            # Cut down the candidates to num_qa_examples.
            sample_shots = sample_shots[:num_qa_examples]
            qa_shots.append(sample_shots)

    return qa_shots


def create_knowledge_generation_examples(
    retriever_output: Union[str, List[Dict[str, Any]]],
    question_to_explanations: Dict[str, Dict[str, List[str]]],
    num_kg_examples: Optional[int]=5,
    add_negative_explanations: Optional[bool]=False,
) -> List[List[Dict[str, Any]]]:
    """
    Create examples for the knowledge generation task using the retriever output and explanations.

    Parameters:
    - retriever_output (Union[str, List[Dict[str, Any]]]): The path to the retriever output file or a list of dictionaries containing the retriever output.
    - question_to_explanations (Dict[str, Dict[str, List[str]]]): A dictionary mapping question IDs to their corresponding positive and negative explanations.
    - num_kg_examples (Optional[int]): The number of knowledge generation examples to generate per sample.
    - add_negative_explanations (Optional[bool]): Whether to include negative explanations.

    Returns:
    - List[List[Dict[str, Any]]]: A list of lists containing the knowledge generation examples.

    Raises:
    - FileNotFoundError: If the retriever_output is a path and does not exist.
    """
    kg_shots = []
    if isinstance(retriever_output, str):
        if not os.path.exists(retriever_output):
            raise FileNotFoundError(
                f"The retriever output path {retriever_output} does not exist. Please provide a valid path."
            )
        with open(retriever_output, "r") as retriever_output:
            retriever_output = [json.loads(line) for line in retriever_output]
    for line in retriever_output:
        sample_shots = []

        for candidate in line["candidates"]:
            example_id = candidate["doc"]["id"]
            example_question = candidate["doc"]["contents"].split(" [SEP] ")[0]
            example_choices = candidate["doc"]["contents"].split(" [SEP] ")[1:]
            example_choices = [
                {
                    "label": choice.split(".")[0].strip(),
                    "text": choice.split(".")[1].strip(),
                }
                for choice in example_choices
            ]

            # Get the answer from the metadata.
            example_metadata = candidate["doc"]["metadata"]
            example_answer = example_metadata["answerKey"]

            # Get the explanations for the example.
            example_knowledge = question_to_explanations[example_id]["positives"]
            if add_negative_explanations:
                example_knowledge.extend(
                    question_to_explanations[example_id]["negatives"]
                )

            if not example_knowledge:
                continue

            sample_shots.append(
                {
                    "id": example_id,
                    "question": example_question,
                    "choices": example_choices,
                    "knowledge": example_knowledge,
                    "answer": example_answer,
                }
            )

        sample_shots = sample_shots[:num_kg_examples]
        kg_shots.append(sample_shots)

    return kg_shots
