from typing import List, Union, Optional
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
        model_name: Union[str, PreTrainedModel], 
        tokenizer_name: Optional[Union[str, PreTrainedTokenizer]]=None, 
        device: Optional[str]='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> tuple:
    """
    Load model and tokenizer from the given model and tokenizer names.

    Parameters:
    - model_name (Union[str, PreTrainedModel]): Model name or PreTrainedModel instance.
    - tokenizer_name (Optional[Union[str, PreTrainedTokenizer]]): Tokenizer name or PreTrainedTokenizer instance. If None, the model name will be used.
    - device (str): Device to load the model onto (default is 'cuda' if available, otherwise 'cpu').

    Returns:
    - tuple: A tuple containing the loaded model (AutoModelForCausalLM) and tokenizer (AutoTokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer_name is None:
        logger.info("Tokenizer name is not provided. Using model name instead.")
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True, padding_side="left"
    )

    if tokenizer.pad_token_id is None and tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.pad_token_id is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def tokenizer_handler(
        inputs: torch.Tensor, 
        tokenizer: AutoTokenizer,
    ) -> torch.Tensor:
    """
    Handle the tokenizer inputs.

    Parameters:
    - inputs (torch.Tensor): Inputs.
    - tokenizer (AutoTokenizer): Tokenizer.

    Returns:
    - torch.Tensor: Processed inputs.
    """
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
    return inputs

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: List[str],
    max_new_tokens: Optional[int]=256,
    device: Optional[str]='cuda' if torch.cuda.is_available() else 'cpu',
) -> tuple:
    """
    Generate text using the model.

    Parameters:
    - model (AutoModelForCausalLM): Model.
    - tokenizer (AutoTokenizer): Tokenizer.
    - prompt (List[str]): List of conversation turns.
    - max_new_tokens (int): Maximum number of new tokens.
    - device (str): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

    Returns:
    - tuple: A tuple containing the model outputs and the generated text.
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
    inputs = tokenizer_handler(inputs, tokenizer)

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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: List[str],
    labels: Optional[List[str]]=["A", "B", "C", "D", "E"],
    max_new_tokens: Optional[int]=1,
    return_scores: Optional[bool]=False,
    device: Optional[str]="cuda" if torch.cuda.is_available() else "cpu",
) -> Union[str, tuple]:
    """
    Get the answer from the model.

    Parameters:
    - model (AutoModelForCausalLM): Model.
    - tokenizer (AutoTokenizer): Tokenizer.
    - prompt (List[str]): List of conversation turns.
    - labels (Optional[List[str]]): Labels, default is ["A", "B", "C", "D", "E"].
    - max_new_tokens (Optional[int]): Maximum number of new tokens.
    - return_scores (Optional[bool]): Return scores.
    - device (Optional[str]): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

    Returns:
    - str: Answer (one of the labels).
    - Optional[torch.Tensor]: Scores (if return_scores is True).
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


def get_model_knowledge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: List[str],
    max_generated_knowledge: Optional[bool]=None,
    max_new_tokens: Optional[int]=256,
    device: Optional[str]="cuda" if torch.cuda.is_available() else "cpu",
) -> List[str]:
    """
    Generate knowledge using the model.

    Parameters:
    - model (AutoModelForCausalLM): Model.
    - tokenizer (AutoTokenizer): Tokenizer.
    - prompt (List[str]): List of conversation turns.
    - max_generated_knowledge (Optional[bool]): Maximum number of knowledge entries to generate.
    - max_new_tokens (Optional[int]): Maximum number of new tokens.
    - device (Optional[str]): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

    Returns:
    - List[str]: Generated knowledge.
    """
    # Generate text using the model.
    _, generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Process the generated text.
    knowledge = generated_text.split("\n\n")[0].split("\n")
    knowledge = [k.replace("*", "") for k in knowledge]
    knowledge = [k.strip() for k in knowledge]
    knowledge = [k for k in knowledge if k]
    knowledge = knowledge[:max_generated_knowledge]

    return knowledge
