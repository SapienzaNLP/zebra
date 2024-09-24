# Templates for the system instructions.
ARC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and up to 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, up to 5 options (labeled A, B, C, D, and E), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

CSQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 5 options (labeled A, B, C, D, and E), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

CSQA2 = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

OBQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 4 options (labeled A, B, C, and D). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 4 options (labeled A, B, C, and D), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

PIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete and 2 possible solutions (labeled A and B). \
Your task is to choose the label corresponding to the best solution. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a questionor a text to complete, 2 possible solutions (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best solution based on the given explanations. \
Do you understand the task?\
""",
}

QASC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 8 options (labeled A, B, C, D, E, F, G and H). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 8 options (labeled A, B, C, D, E, F, G and H), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

SIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 3 options (labeled A, B and C). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 3 options (labeled A, B and C), and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

WG = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question where one word has been replaced with \"_\" and 2 options (labeled A and B) to replace \"_\". \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B) to replace \"_\", and a list of explanations. \
Your task is to choose the label corresponding to the best answer for the question based on the given explanations. \
Do you understand the task?\
""",
}

# Templates for the fewshot examples.
MCQ_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}\
"""

MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}

Explanations:
{knowledge}\
"""

SHOT_TEMPLATES = {
    "mcq": MCQ_EXAMPLE_TEMPLATE,
    "mcq_with_kg": MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE,
}

# Mapping from dataset names to their respective instruction prompts.
DATASET_TAGS = {
    "arc": ARC,
    "csqa": CSQA,
    "csqa2": CSQA2,
    "obqa": OBQA,
    "piqa": PIQA,
    "qasc": QASC,
    "siqa": SIQA,
    "wg": WG,
}
