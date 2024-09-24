from typing import Dict, List

from loguru import logger


def compute_metrics(
    ground_truths: List, answers_without_knowledge: List, answers_with_knowledge: List
) -> Dict[str, float]:
    """
    Compute metrics.

    Args:
        ground_truths (List): Ground truth answers.
        answers (List): Answers without knowledge.
        answers_with_knowledge (List): Answers with knowledge.

    Returns:
        results (Dict): Results.
    """
    correct_without_knowledge = 0
    correct_with_knowledge = 0
    for gt, a, a_k in zip(
        ground_truths,
        answers_without_knowledge,
        answers_with_knowledge,
    ):
        if gt == a:
            correct_without_knowledge += 1
        if gt == a_k:
            correct_with_knowledge += 1

    accuracy_without_knowledge = correct_without_knowledge / len(ground_truths)
    accuracy_with_knowledge = correct_with_knowledge / len(ground_truths)

    results = {
        "accuracy_without_knowledge": accuracy_without_knowledge,
        "accuracy_with_knowledge": accuracy_with_knowledge,
    }

    # Print the results.
    logger.info(f"Accuracy without knowledge: {accuracy_without_knowledge}")
    logger.info(f"Accuracy with knowledge: {accuracy_with_knowledge}")

    return results
