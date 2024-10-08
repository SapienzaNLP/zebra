from typing import List

from zebra.prompts import prompts

DATASET_TAGS = prompts.DATASET_TAGS
MCQ_EXAMPLE_TEMPLATE = prompts.MCQ_EXAMPLE_TEMPLATE
MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE = prompts.MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE
SHOT_TEMPLATES = prompts.SHOT_TEMPLATES


def prepare_sample_for_mcq(
    sample,
    dataset_tag: str,
    template_name: str = "mcq",
    sample_knowledge: List[str] = [],
) -> str:
    """
    Prepare the prompt for the model.

    Args:
        sample (Dict): Sample with the question and the options.
        template_name (str): Template for the prompt; one of ["mcq", "mcq_with_kg"].
        sample_knowledge (List[str]): Explanations for the options of the sample.

    Returns:
        prompt (List[str]): Prompt for the model. The prompt consists of the conversation turns (user and assistant turns).
    """

    # Get the template.
    template = SHOT_TEMPLATES[template_name]

    # Prepare the sample.
    question = sample["question"]["stem"]
    choices = sample["question"]["choices"]
    choices = [f"* {c['label']}: {c['text']}".strip() for c in choices]
    choices = "\n".join(choices)

    # Preprocess the sample knowledge to display it in the prompt.
    # Each explanation is displayed as "* {explanation}".
    if sample_knowledge is not None:
        sample_knowledge = [k if k.endswith(".") else f"{k}." for k in sample_knowledge]
        sample_knowledge = [f"* {k.strip()}" for k in sample_knowledge]
        sample_knowledge = "\n".join(sample_knowledge)

    # Prepare the final shot for the user turn.
    final_shot = template.format(
        question=question,
        knowledge=sample_knowledge,
        choices=choices,
    )

    # Prepare the assistant turn.
    assistant_reply = "Answer: "

    # Build the conversation turns.
    prompt = [
        DATASET_TAGS[dataset_tag][template_name],
        "Yes, I understand. Please provide the question and the possible options.",
    ] + [final_shot, assistant_reply]

    # Return the prompt / conversation turns.
    return prompt
