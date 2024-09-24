import argparse
import json
import os
import random

import jsonlines
import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set the seed for reproducibility.
random.seed(0)
torch.manual_seed(0)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args (Namespace): Command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add arguments.
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        required=True,
        help="Model name or path.",
    )
    parser.add_argument(
        "--tokenizer_name",
        "-tn",
        type=str,
        required=True,
        help="Tokenizer name or path.",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        required=True,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Output directory for the predictions.",
    )
    parser.add_argument(
        "--num_return_sequences",
        "-nrs",
        type=int,
        default=None,
        help="Maximum number of knowledge to generate.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="Device (default is 'cuda').",
    )
    # Parse arguments.
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse arguments.
    args = parse_arguments()

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name, device_map=args.device, trust_remote_code=True
    )

    model.eval()

    with open(args.data_path, "r") as f_in:
        data = f_in.readlines()

    # Get output path.
    dataset_tag = args.data_path.split("/")[-1].split(".")[0]
    output_filename = f'{dataset_tag}|{args.model_name.split("/")[1]}|explanations=rainier|num_return_sequences={args.num_return_sequences}.jsonl'
    output_path = os.path.join(args.output_dir, output_filename)

    # Create output directory if it does not exist.
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with jsonlines.open(output_path, "w") as fout:
        # Iterate over the samples.
        iterator = tqdm.tqdm(
            enumerate(data),
            "Generating knowledge...",
        )

        for i, sample in iterator:
            sample = json.loads(sample)

            # Get information about the sample.
            sample_id = sample["id"]
            question = sample["question"]["stem"]
            choices = sample["question"]["choices"]
            ground_truth = sample["answerKey"]

            # Format the input text.
            input_text = f"{question} \\n " + " ".join(
                [f"({choice['label']}) {choice['text']}" for choice in choices]
            )

            # Generate knowledge.
            input = tokenizer(input_text, return_tensors="pt").to(args.device)
            output_ids = model.generate(
                input["input_ids"],
                do_sample=True,
                top_p=0.5,
                num_return_sequences=args.num_return_sequences,
            )
            knowledge = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            knowledge = [k.strip() for k in knowledge]
            sample["knowledge"] = knowledge

            # Write the sample to the output file.
            fout.write(sample)
