import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str) -> tuple:
    """
    Load model and tokenizer from the given model name.

    Args:
        model_name (str): Model name or path.

    Returns:
        model (AutoModelForCausalLM): Model.
        tokenizer (AutoTokenizer): Tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )

    if tokenizer.pad_token_id is None and tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.pad_token_id is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
    device="cuda",
) -> tuple:
    """
    Generate text using the model.

    Args:
        model (AutoModelForCausalLM): Model.
        tokenizer (AutoTokenizer): Tokenizer.
        prompt (str): List of conversation turns.
        max_new_tokens (int): Maximum number of new tokens.
        device (str): Device (default is "cuda").

    Returns:
        model_outputs (Dict): Model outputs.
    """

    # Build the conversation turns in the format required by chat templates.
    messages = []
    for turn_id, turn in enumerate(prompt):
        if turn_id % 2 == 0:
            messages.append({"role": "user", "content": turn})
        else:
            messages.append({"role": "assistant", "content": turn})

    # Apply the chat template to the messages.
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Remove the special tokens added by the tokenizer at the end of the input.
    if tokenizer.__class__.__name__ == "Phi3SmallTokenizer":
        inputs = inputs[:, :-3]
    elif tokenizer.__class__.__name__ == "GemmaTokenizerFast":
        inputs = inputs[:, :-2]
    elif tokenizer.__class__.__name__ == "Qwen2TokenizerFast":
        inputs = inputs[:, :-2]
    elif (
        tokenizer.__class__.__name__ == "LlamaTokenizerFast"
        and tokenizer.name_or_path == "microsoft/Phi-3-mini-128k-instruct"
    ):
        inputs = inputs[:, :-2]
    elif inputs[0][-1].item() in tokenizer.added_tokens_decoder:
        inputs = inputs[:, :-1]

    # Create the attention mask.
    attention_mask = torch.ones_like(inputs).to(device)

    # Generate text using the model.
    model_outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated text.
    generated_text = tokenizer.decode(
        model_outputs.sequences[0][inputs[0].shape[0] :],
        skip_special_tokens=True,
    )

    return model_outputs, generated_text


def get_model_answer(
    model,
    tokenizer,
    prompt,
    labels=["A", "B", "C", "D", "E"],
    max_new_tokens=1,
    return_scores=False,
    device="cuda",
) -> str:
    """
    Get the answer from the model.

    Args:
        model (AutoModelForCausalLM): Model.
        tokenizer (AutoTokenizer): Tokenizer.
        prompt (str): Prompt.
        labels (List[str]): Labels, default is ["A", "B", "C", "D", "E"].
        max_new_tokens (int): Maximum number of new tokens.
        return_scores (bool): Return scores.

    Returns:
        answer (str): Answer (one of the labels).
        scores (torch.Tensor): Scores (if return_scores is True).
    """
    # Generate alternative labels with whitespaces in front.
    labels.extend([f" {label}" for label in labels])

    # Generate text using the model.
    model_outputs, _ = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Get the probabilities of the first token.
    probabilities = torch.softmax(model_outputs.scores[-1], dim=-1)[0]

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]

    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]

    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    answer = answer.lstrip()

    if return_scores:
        return answer, probabilities
    else:
        return answer
