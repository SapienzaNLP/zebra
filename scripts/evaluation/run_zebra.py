import argparse
import csv
import json
import os
import random

import torch
import tqdm
from zebra.utils.data_utils import (create_knowledge_generation_examples,
                              create_qa_fewshot_examples, load_evaluation_data,
                              load_explanations)
from zebra.utils.metric_utils import compute_metrics
from zebra.utils.model_utils import (get_model_answer, get_model_knowledge,
                               load_model_and_tokenizer)
from zebra.utils.prompt_utils import (prepare_sample_for_knowledge_generation,
                                prepare_sample_for_mcq)
from zebra.utils.stdout_utils import (get_output_path, write_header, write_row,
                                write_scores)

UPSAMPLING_FACTOR = 5

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
        "--retriever_output_path",
        "-rop",
        type=str,
        required=True,
        help="Path to the retriever output.",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        required=True,
        help="Path to the dataset to use for evaluation.",
    )
    parser.add_argument(
        "--explanations_path",
        "-ep",
        type=str,
        required=True,
        help="Path to the dataset with the explanations. Can be either a local path or a Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--dataset_tag",
        "-dt",
        type=str,
        required=True,
        help="Name of the evaluation dataset.",
        choices=["arc", "csqa", "csqa2", "obqa", "piqa", "qasc", "wg"],
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Output directory for the predictions and the scores.",
    )

    # Add optional arguments.
    parser.add_argument(
        "--fewshot_data_path",
        "-fdp",
        type=str,
        help="Path to the dataset to use for fewshot learning. This only applies for the question answering step.",
    )
    parser.add_argument(
        "--explanations_split",
        "-es",
        type=str,
        required=not os.path.exists(parser.parse_known_args()[0].explanations_path),
        choices=[
            "all",
            "csqa-train-ecqa",
            "arc-train-gemini",
            "csqa-train-gemini",
            "csqa2-train-gemini",
            "obqa-train-gemini",
            "piqa-train-gemini",
            "qasc-train-gemini",
            "wg-train-gemini",
        ],
        help="Name of the explanations split. If the explanations are loaded from a Hugging Face dataset, this argument is required.",
    )
    parser.add_argument(
        "--plain",
        "-p",
        action="store_true",
        default=False,
        help="Run the model without external knowledge.",
    )
    parser.add_argument(
        "--oracle",
        "-o",
        action="store_true",
        default=False,
        help="Run the model with oracle knowledge.",
    )
    parser.add_argument(
        "--examples",
        "-e",
        action="store_true",
        default=False,
        help="Run model with knowledge adapted from retrieved examples.",
    )
    parser.add_argument(
        "--max_generated_knowledge",
        "-mgk",
        type=int,
        default=None,
        help="Maximum number of knowledge statements to generate.",
    )
    parser.add_argument(
        "--add_negative_explanations",
        "-ane",
        action="store_true",
        help="Add the negative explanations to the positive explanations.",
    )
    parser.add_argument(
        "--num_kg_examples",
        "-nkge",
        type=int,
        default=1,
        help="Maximum number of retrieved examples to be used for knowledge generation.",
    )
    parser.add_argument(
        "--num_qa_examples",
        "-nqae",
        type=int,
        default=0,
        help="Maximum number of fewshot examples to be used for question answering.",
    )
    parser.add_argument(
        "--limit_samples",
        "-ls",
        type=int,
        default=None,
        help="Limit the number of samples to be processed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: 'cuda').",
    )

    # Parse arguments.
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments.
    args = parse_arguments()

    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name, 
        device=args.device
    )

    # Load datasets.
    eval_data = load_evaluation_data(
        data_path=args.data_path, 
        limit_samples=args.limit_samples
    )

    # Load local or remote explanations.
    question_to_explanations = load_explanations(
        explanations_path=args.explanations_path,
        explanations_split=args.explanations_split
    )

    # Generate random examples for fewshot question answering.
    qa_shots = create_qa_fewshot_examples(
        len_eval_data=len(eval_data),
        question_to_explanations=question_to_explanations,
        fewshot_data_path=args.fewshot_data_path,
        num_qa_examples=args.num_qa_examples,
        upsampling_factor=UPSAMPLING_FACTOR,
        add_negative_explanations=args.add_negative_explanations,
    )

    # Fetch retrieved examples for the knowledge generation step.
    kg_shots = create_knowledge_generation_examples(
        retriever_output=args.retriever_output_path,
        num_kg_examples=args.num_kg_examples,
        question_to_explanations=question_to_explanations,
        add_negative_explanations=args.add_negative_explanations,
    )

    # Initialize result lists.
    ground_truths = []
    answers_without_knowledge = []
    answers_with_knowledge = []
    answers_with_oracle_knowledge = []

    # Get output path.
    output_path, scores_path = get_output_path(
        data_path=args.data_path,
        model_name=args.model_name,
        num_kg_examples=args.num_kg_examples,
        num_qa_examples=args.num_qa_examples,
        add_negative_explanations=args.add_negative_explanations,
        explanations_path=args.explanations_path,
        explanations_split=args.explanations_split,
        retriever_output_path=args.retriever_output_path,
        output_dir=args.output_dir,
    )

    # Write the output to the file.
    with open(output_path, "w") as fout:

        writer = csv.writer(fout, delimiter="\t")
        write_header(writer)

        # Iterate over the samples.
        iterator = tqdm.tqdm(
            enumerate(zip(eval_data, qa_shots, kg_shots)),
            total=len(eval_data),
            desc="Generating answers...",
        )

        for i, (sample, random_examples, retrieved_examples) in iterator:
            sample = json.loads(sample)

            # Get information about the sample.
            sample_id = sample["id"]
            question = sample["question"]["stem"]
            choices = sample["question"]["choices"]
            labels = [choice["label"] for choice in choices]

            # Get the ground truth.
            ground_truth = sample["answerKey"]
            ground_truths.append(ground_truth)

            # Get the oracle knowledge.
            oracle_knowledge = question_to_explanations[sample_id]["positives"]
            if args.add_negative_explanations:
                oracle_knowledge.extend(
                    question_to_explanations[sample_id]["negatives"]
                )

            answer_without_knowledge = None
            answer_with_knowledge = None
            answer_with_oracle_knowledge = None
            generated_knowledge = []

            if args.plain:
                # Prepare the prompt for the model.
                prompt = prepare_sample_for_mcq(
                    sample=sample,
                    examples=random_examples,
                    dataset_tag=args.dataset_tag,
                    template_name="mcq",
                    use_example_knowledge=False,
                    sample_knowledge=None,
                )

                # Get the answer from the model.
                answer_without_knowledge = get_model_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    labels=labels,
                    max_new_tokens=1,
                    return_scores=False,
                    device=args.device,
                )

            if args.oracle:
                # Prepare the prompt for the model.
                prompt_with_oracle_knowledge = prepare_sample_for_mcq(
                    sample=sample,
                    examples=random_examples,
                    dataset_tag=args.dataset_tag,
                    template_name="mcq_with_kg",
                    use_example_knowledge=True,
                    sample_knowledge=oracle_knowledge,
                )

                # Get the answer from the model.
                answer_with_oracle_knowledge = get_model_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_with_oracle_knowledge,
                    labels=labels,
                    max_new_tokens=1,
                    return_scores=False,
                    device=args.device,
                )

            if args.examples:
                # Prepare the prompt for knowledge generation.
                prompt_for_knowledge_gen = prepare_sample_for_knowledge_generation(
                    sample=sample,
                    shot_samples=retrieved_examples,
                    dataset_tag=args.dataset_tag,
                )

                # Get the generated knowledge.
                generated_knowledge = get_model_knowledge(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_for_knowledge_gen,
                    max_generated_knowledge=args.max_generated_knowledge,
                    max_new_tokens=256,
                    device=args.device,
                )

                # Prepare the prompt for the model.
                prompt_with_knowledge = prepare_sample_for_mcq(
                    sample=sample,
                    sample_knowledge=generated_knowledge,
                    examples=random_examples,
                    dataset_tag=args.dataset_tag,
                    template_name="mcq_with_kg",
                    use_example_knowledge=True,
                )

                # Get the answer from the model.
                answer_with_knowledge = get_model_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_with_knowledge,
                    labels=labels,
                    max_new_tokens=1,
                    return_scores=False,
                    device=args.device,
                )

            # Add the answers to the list.
            answers_without_knowledge.append(answer_without_knowledge)
            answers_with_knowledge.append(answer_with_knowledge)
            answers_with_oracle_knowledge.append(answer_with_oracle_knowledge)

            # Write the row to the file.
            write_row(
                writer=writer,
                question=question,
                retrieved_examples=retrieved_examples,
                choices=choices,
                ground_truth=ground_truth,
                oracle_knowledge=oracle_knowledge,
                generated_knowledge=generated_knowledge,
                answer_without_knowledge=answer_without_knowledge,
                answer_with_knowledge=answer_with_knowledge,
                answer_with_oracle_knowledge=answer_with_oracle_knowledge,
                num_kg_examples=args.num_kg_examples,
            )

    # Compute the metrics.
    metrics = compute_metrics(
        ground_truths=ground_truths,
        answers_without_knowledge=answers_without_knowledge,
        answers_with_knowledge=answers_with_knowledge,
        answers_with_oracle_knowledge=answers_with_oracle_knowledge,
    )

    # Write the scores to a file.
    write_scores(
        scores_path=scores_path,
        metrics=metrics,
    )
