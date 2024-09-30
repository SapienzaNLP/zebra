import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, List
import os
import jsonlines

random.seed(42)

# Here you can find ad example showing the required dataset format:
# NOTE: negatives and hard negatives are not explicitly required, they will be generated at training time.

# [
# {
#     "question": "....",
#     "answers": ["...", "...", "..."],
#     "positive_ctxs": [{
#         "title": "...",
#         "text": "...."
#     }],
#     "negative_ctxs": ["..."],
#     "hard_negative_ctxs": ["..."]
# },
# ...
# ]


def compute_permutations(
    question: str, 
    choices: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Compute permutations of the labels for the choices of a question.

    Parameters:
    - question (str): The question.
    - choices (List[Dict[str, str]]): The choices of the question.

    Returns:
    - permuted_positive_ctxs (List[Dict[str, str]]): The permuted positive contexts.
    """
    permuted_positive_ctxs = []
    label_choices = [choice["label"] for choice in choices]
    text_choices = [choice["text"] for choice in choices]
    permuted_text_choices = list(permutations(text_choices))

    for permuted_text_choice in permuted_text_choices:
        permuted_choice = " [SEP] ".join(
            [f"{lc}. {pc}" for lc, pc in zip(label_choices, permuted_text_choice)]
        )
        permuted_positive_ctxs.append({"text": f"{question} [SEP] {permuted_choice}"})

    return permuted_positive_ctxs


def compute_subsets(
    question: str, 
    choices: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Compute subsets of the labels for the choices of a question.

    Parameters:
    - question (str): The question.
    - choices (List[Dict[str, str]]): The choices of the question.

    Returns:
    - subset_positive_ctxs (List[Dict[str, str]]): The subset positive contexts.
    """
    subset_positive_ctxs = []
    label_choices = [choice["label"] for choice in choices]
    text_choices = [choice["text"] for choice in choices]

    for r in range(1, len(text_choices) + 1):
        subsets = combinations(text_choices, r)
        for subset in subsets:
            subset_choice = " [SEP] ".join(
                [f"{lc}. {sc}" for lc, sc in zip(label_choices, subset)]
            )
            subset_positive_ctxs.append({"text": f"{question} [SEP] {subset_choice}"})

    return subset_positive_ctxs


def random_sample(
    positive_ctxs: List[Dict[str, str]], 
    amount: int
) -> List[Dict[str, str]]:
    """
    Randomly sample a subset of positive contexts.

    Parameters:
    - positive_ctxs (List[Dict[str, str]]): The positive contexts.
    - amount (int): The amount of positive contexts to sample.

    Returns:
    - sampled_ctxs (List[Dict[str, str]]): The sampled positive contexts.
    """
    if len(positive_ctxs) < amount:
        return positive_ctxs
    sampled_ctxs = random.sample(positive_ctxs, amount)
    return sampled_ctxs


def find_examples(
    concept_to_questions: Dict[str, List[str]], 
    question: str, 
    concept: str
) -> List[Dict[str, str]]:
    """
    Find similar examples based on the concept of the question.

    Parameters:
    - concept_to_questions (Dict[str, List[str]]): The concept to questions mapping.
    - question (str): The question.
    - concept (str): The concept of the question.

    Returns:
    - similar_examples (List[Dict[str, str]]): The similar examples.
    """
    similar_examples = []
    for k, v in concept_to_questions.items():
        if concept in k:
            for sample in v:
                if question not in sample["question"]["stem"]:
                    sample_question = sample["question"]["stem"]
                    sample_choices = sample["question"]["choices"]
                    joined_sample_choices = " [SEP] ".join(
                        [
                            f"{choice['label']}. {choice['text']}"
                            for choice in sample_choices
                        ]
                    )
                    similar_examples.append(
                        {"text": f"{sample_question} [SEP] {joined_sample_choices}"}
                    )
    return similar_examples


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
        help="Path to the data file to be used to create the data for training or validation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the generated data.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--find_examples",
        action="store_true",
        help="Whether to find similar examples based on the concept of the question.",
    )
    parser.add_argument(
        "--examples_amount",
        type=int,
        default=32,
        help="Maximum amount of similar examples to find based on the concept of the question.",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Whether to augment the training data by computing permutations and subsets of the labels.",
    )
    parser.add_argument(
        "--augmentation_amount",
        type=int,
        default=32,
        help="Maximum amount of augmented examples to generate.",
    )

    # Parse arguments.
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with jsonlines.open(args.output_path, "w") as writer:

        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"The data path {args.data_path} does not exist. Please provide a valid path.")
        
        with open(args.data_path, "r") as reader:
            samples = reader.readlines()
            
        concept_to_questions = defaultdict(list)
        ## Create a dictionary of questions (v) based on the concept (k)
        for line in samples:
            line = json.loads(line)
            concept_to_questions[line["question"]["question_concept"]].append(line)
        ## Iterate over the samples and create training lines
        for line in samples:
            line = json.loads(line)
            train_line = {}
            ## The trainin example will be in the form of: Q [SEP] A [SEP] B [SEP] C [SEP] D ...
            question = line["question"]["stem"]
            choices = line["question"]["choices"]
            joined_choices = " [SEP] ".join(
                [f"{choice['label']}. {choice['text']}" for choice in choices]
            )
            train_line["question"] = f"{question} [SEP] {joined_choices}"
            positive_ctxs = []
            ## Compute similar questions based on the concept
            if args.find_examples:
                similar_questions = find_examples(
                    concept_to_questions,
                    question,
                    concept=line["question"]["question_concept"],
                )
                similar_questions = random_sample(
                    similar_questions, amount=args.examples_amount
                )
                positive_ctxs.extend(similar_questions)
            ## Compute permutations and subsets of the labels to augment the positives for the training question.
            if args.augmentation:
                sample_permutations = compute_permutations(question, choices)
                sample_subsets = compute_subsets(question, choices)
                positive_ctxs.extend(
                    random_sample(
                        sample_permutations + sample_subsets,
                        amount=args.augmentation_amount,
                    )
                )
            ## Write training line
            train_line["positive_ctxs"] = positive_ctxs
            writer.write(train_line)
