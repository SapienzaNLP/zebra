from zebra import Zebra

# Load Zebra with language model, retriever, document index and explanations.
zebra = Zebra(model="meta-llama/Meta-Llama-3-8B-Instruct")

# Provide a question and answer choices.
questions = [
    "What should you do if you see someone hurt and in need of help?",
    "If your friend is upset, what is the best way to support them?",
    "What should you do if your phone battery is running low in a public place?",
    "What should you do if you are running late for an important meeting?",
]

choices = [
    ["Walk away.", "Call for help.", "Take a photo for social media."],
    ["Listen to them and offer comfort.", "Tell them they are overreacting.", "Ignore them and walk away."],
    ["Borrow a stranger's phone.", "Use public charging station.", "Leave your phone unattended while it charges."],
    ["Rush through traffic.", "Call and inform them you will be late.", "Do not show up at all."],
]

# Generate knowledge and perform question answering.
zebra_output = zebra.pipeline(questions=questions, choices=choices, return_dict=True)
print()