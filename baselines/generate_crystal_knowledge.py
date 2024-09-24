import argparse
import json
import os
import random

import jsonlines
import torch
import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "crystal-11b" in args.model_name:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            device_map=args.device,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            device_map=args.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()

    with open(args.data_path, "r") as f_in:
        data = f_in.readlines()

    # Get output path.
    dataset_tag = args.data_path.split("/")[-1].split(".")[0]
    output_filename = f'{dataset_tag}|{args.model_name.split("/")[1]}|explanations=crystal|num_return_sequences={args.num_return_sequences}.jsonl'
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

            max_question_len, max_knowledge_len, max_answer_len = 128, 32, 2
            top_p = 0.5

            prompt = input_text + " \\n Knowledge: "
            prompt_tok = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation="longest_first",
                max_length=max_question_len,
            ).to(
                args.device
            )  # (1, QL)
            knowledges_ids = model.generate(
                input_ids=prompt_tok.input_ids,
                attention_mask=prompt_tok.attention_mask,
                max_length=max_knowledge_len + 1,
                min_length=3,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                top_p=top_p,
            )  # (K, KL); begins with 0 ([BOS]); ends with 1 ([EOS])

            knowledges_ids = knowledges_ids[
                :, 1:
            ].contiguous()  # no beginning; ends with 1 ([EOS])
            knowledge = tokenizer.batch_decode(
                knowledges_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            knowledge = [k.strip() for k in knowledge]
            sample["knowledge"] = knowledge

            # Write the sample to the output file.
            fout.write(sample)
