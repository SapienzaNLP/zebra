from typing import Dict, Optional, List

from itertools import repeat
from loguru import logger


def compute_metrics(
    ground_truths: List[str],
    answers_without_knowledge: Optional[List[str]]=None,
    answers_with_knowledge: Optional[List[str]]=None,
    answers_with_oracle_knowledge: Optional[List[str]]=None,
) -> Dict[str, float]:
    """
    Compute metrics (accuracy) for the answers.

    Parameters:
    - ground_truths (List[str]): Ground truth answers.
    - answers_without_knowledge (Optional[List[str]]): Answers without knowledge.
    - answers_with_knowledge (Optional[List[str]]): Answers with knowledge.
    - answers_with_oracle_knowledge (Optional[List[str]]): Answers with oracle knowledge.

    Returns:
    - results (Dict[str, float]): Dictionary containing accuracy metrics.
    """
    if answers_without_knowledge is None:
        answers_without_knowledge = list(repeat(None, len(ground_truths)))
    if answers_with_knowledge is None:
        answers_with_knowledge = list(repeat(None, len(ground_truths)))
    if answers_with_oracle_knowledge is None:
        answers_with_oracle_knowledge = list(repeat(None, len(ground_truths)))
    correct_without_knowledge = 0
    correct_with_knowledge = 0
    correct_with_oracle_knowledge = 0
    for (
        gt,
        a,
        a_k,
        a_o_k,
    ) in zip(
        ground_truths,
        answers_without_knowledge,
        answers_with_knowledge,
        answers_with_oracle_knowledge,
    ):
        if gt == a:
            correct_without_knowledge += 1
        if gt == a_k:
            correct_with_knowledge += 1
        if gt == a_o_k:
            correct_with_oracle_knowledge += 1

    accuracy_without_knowledge = correct_without_knowledge / len(ground_truths)
    accuracy_with_knowledge = correct_with_knowledge / len(ground_truths)
    accuracy_with_oracle_knowledge = correct_with_oracle_knowledge / len(ground_truths)

    results = {
        "accuracy_without_knowledge": accuracy_without_knowledge,
        "accuracy_with_knowledge": accuracy_with_knowledge,
        "accuracy_with_oracle_knowledge": accuracy_with_oracle_knowledge,
    }

    # Print the results.
    logger.info(f"Accuracy without knowledge: {accuracy_without_knowledge}")
    logger.info(f"Accuracy with knowledge: {accuracy_with_knowledge}")
    logger.info(f"Accuracy with oracle knowledge: {accuracy_with_oracle_knowledge}")

    return results
