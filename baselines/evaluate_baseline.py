import argparse
import csv
import json
import random

import torch
import tqdm
from utils.data_utils import load_evaluation_data
from utils.metric_utils import compute_metrics
from utils.model_utils import get_model_answer, load_model_and_tokenizer
from utils.prompt_utils import prepare_sample_for_mcq
from utils.stdout_utils import (get_output_path, write_header, write_row,
                                write_scores)

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
        help="Path to the dataset to use for the evaluation.",
    )
    parser.add_argument(
        "--dataset_tag",
        "-dt",
        type=str,
        required=True,
        help="Name of the evaluation dataset.",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Output directory for the predictions and the scores.",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        required=True,
        choices=["selftalk", "gkp", "rainier", "crystal", "rbr"],
        help="Run model with the knowledge from one of the available baselines (selftalk, gkp, rainier, crystal, rbr).",
    )

    # Add optional arguments.
    parser.add_argument(
        "--plain",
        "-p",
        action="store_true",
        default=False,
        help="Run model without external knowledge.",
    )
    parser.add_argument(
        "--max_knowledge",
        "-mk",
        type=int,
        default=None,
        help="Maximum number of knowledge statements to use.",
    )
    parser.add_argument(
        "--limit_samples",
        "-ls",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate.",
    )
    # Parse arguments.
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse arguments.
    args = parse_arguments()

    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Load datasets.
    eval_data = load_evaluation_data(
        data_path=args.data_path,
        limit_samples=args.limit_samples,
    )

    # Get output path.
    output_path, scores_path = get_output_path(
        data_path=args.data_path,
        model_name=args.model_name,
        baseline=args.baseline,
        max_knowledge=args.max_knowledge,
        output_dir=args.output_dir,
    )

    ground_truths = []
    answers_without_knowledge = []
    answers_with_knowledge = []

    # Write the output to the file.
    with open(output_path, "w") as fout:

        writer = csv.writer(fout, delimiter="\t")
        write_header(writer)

        # Iterate over the samples.
        iterator = tqdm.tqdm(
            enumerate(eval_data),
            total=len(eval_data),
            desc="Generating answers...",
        )

        for i, sample in iterator:
            sample = json.loads(sample)

            # Get information about the sample.
            sample_id = sample["id"]
            question = sample["question"]["stem"]
            choices = sample["question"]["choices"]
            labels = [choice["label"] for choice in choices]

            # Get the knowledge for the sample and limit the number of knowledge statements.
            if "knowledge" not in sample:
                raise KeyError(
                    """
                    The sample does not contain a 'knowledge' field. Each sample in the evaluation data should contain a 'knowledge' field.
                    The content of the 'knowledge' field should be a list of knowledge statements obtained with the relative baseline.
                """
                )
            knowledge = sample["knowledge"][: args.max_knowledge]

            # Get the ground truth.
            ground_truth = sample["answerKey"]
            ground_truths.append(ground_truth)

            answer_without_knowledge = None
            answer_with_knowledge = None
            generated_knowledge = []

            if args.plain:
                # Prepare the prompt for the model.
                prompt = prepare_sample_for_mcq(
                    sample=sample,
                    sample_knowledge=None,
                    dataset_tag=args.dataset_tag,
                    template_name="mcq",
                )

                # Get the answer from the model.
                answer_without_knowledge = get_model_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    labels=labels,
                    max_new_tokens=1,
                    return_scores=False,
                    device="cuda",
                )

            # Prepare the prompt for the model.
            prompt_with_knowledge = prepare_sample_for_mcq(
                sample=sample,
                sample_knowledge=knowledge,
                dataset_tag=args.dataset_tag,
                template_name="mcq_with_kg",
            )

            # Get the answer from the model.
            answer_with_knowledge = get_model_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_with_knowledge,
                labels=labels,
                max_new_tokens=1,
                return_scores=False,
                device="cuda",
            )

            answers_without_knowledge.append(answer_without_knowledge)
            answers_with_knowledge.append(answer_with_knowledge)

            # Write the row to the file.
            write_row(
                writer=writer,
                question=question,
                choices=choices,
                knowledge=knowledge,
                ground_truth=ground_truth,
                answer_without_knowledge=answer_without_knowledge,
                answer_with_knowledge=answer_with_knowledge,
            )

    # Compute the metrics.
    metrics = compute_metrics(
        ground_truths=ground_truths,
        answers_without_knowledge=answers_without_knowledge,
        answers_with_knowledge=answers_with_knowledge,
    )

    # Write the scores to a file.
    write_scores(
        scores_path=scores_path,
        metrics=metrics,
    )
