from typing import List, Union, Dict, Optional, Any, Tuple
import torch
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from goldenretriever.indexers.inmemory import BaseDocumentIndex
from transformers import PreTrainedModel, PreTrainedTokenizer
from goldenretriever.pytorch_modules.model import GoldenRetriever
from goldenretriever.pytorch_modules import RetrievedSample
from zebra.data.objects import ZebraOutput
from zebra.utils.data_utils import load_explanations, create_knowledge_generation_examples
from zebra.utils.model_utils import (get_model_answer, get_model_knowledge,
                                     load_model_and_tokenizer)
from zebra.utils.prompt_utils import (prepare_sample_for_knowledge_generation,
                                prepare_sample_for_mcq)

random.seed(0)
torch.manual_seed(0)

RETRIEVER_PATH = "sapienzanlp/zebra-retriever-e5-base-v2"
DOCUMENT_INDEX_PATH = "sapienzanlp/zebra-kb"
EXPLANATIONS_PATH = "sapienzanlp/zebra-kb-explanations"
EXPLANATIONS_SPLIT = "all"

class Zebra:
    def __init__(
            self, 
            model:Union[str, PreTrainedModel],
            tokenizer:Optional[Union[str, PreTrainedTokenizer]]=None,
            retriever:Optional[Union[str, PreTrainedModel]]=RETRIEVER_PATH, 
            document_index:Union[str, BaseDocumentIndex]=DOCUMENT_INDEX_PATH,
            explanations:Optional[str]=EXPLANATIONS_PATH,
            explanations_split:Optional[str]=EXPLANATIONS_SPLIT,
            device:Optional[str]='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Zebra class for generating knowledge and answering questions.

        Parameters:
        - model (Union[str, PreTrainedModel]): 
            A path to a HuggingFace model ID or a PreTrainedModel object. The model to use for the question answering task.
        - tokenizer (Union[str, PreTrainedTokenizer]): 
            A path to a HuggingFace tokenizer ID or a PreTrainedTokenizer object. The tokenizer to use for the question answering task.
        - retriever (Union[str, PreTrainedModel]):
            A path to a HuggingFace model ID or a PreTrainedModel object. The retriever model to use to fetch relevant question-knowledge pairs for the input.
        - document_index (Union[str, BaseDocumentIndex]):
            A path to a document index file or a BaseDocumentIndex object. The document index to use for the retriever.
        - explanations (str):
            A path to a file containing explanations for the questions. Can be a local file or a HuggingFace dataset ID.
        - explanations_split (Optional[str]):
            The split to use for the explanations. If the 'explanations' parameter points to a HuggingFace dataset ID, this parameter specifies the split to use.
        - device (Optional[str]): 
            The device to use for the computations. Can be 'cuda' or 'cpu'.

        Returns:
        - Zebra: Zebra object.
        """
        # Load model and tokenizer.
        if isinstance(model, str) and (isinstance(tokenizer, str) or tokenizer is None):
            self.log("Loading model and tokenizer.")
            model, tokenizer = load_model_and_tokenizer(
                model_name=model,
                tokenizer_name=tokenizer,
                device=device
            )
        self.model = model
        self.tokenizer = tokenizer
        
        # Load document index.
        self.log("Loading retriever and document index.")
        self.retriever = GoldenRetriever(
            question_encoder=retriever,
            document_index=document_index,
            device=device,
        )
        
        # Load explanations.
        if isinstance(explanations, str):
            self.log("Loading explanations.")
            question_to_explanations = load_explanations(
                explanations_path=explanations, 
                explanations_split=explanations_split
            )

        self.question_to_explanations = question_to_explanations
        self.device = device

    def log(
        self, 
        *messages:Union[str, List[str]],
        type:Optional[str]="info",
        **kwargs
    ):
        """
        Log one or more messages using the logger.
        """
        for message in messages:
            msg_list = message if isinstance(message, list) else [message]
            for msg in msg_list:
                logger_method = getattr(logger, type)
                logger_method(msg, **kwargs)

    def pipeline(
        self,
        questions:Union[str, List[str]],
        choices:Optional[Union[List[str], List[List[str]], List[Dict[str, str]], List[List[Dict[str, str]]]]],
        top_k:Optional[int]=5,
        batch_size:Optional[int]=64,
        num_kg_examples:Optional[int]=5,
        dataset_tag:Optional[str]="custom",
        questions_domain:Optional[str]="commonsense",
        kg_template_name:Optional[str]="knowledge_generation",
        qa_template_name:Optional[str]="mcq_with_kg",
        max_generated_knowledge:Optional[int]=None,
        max_new_tokens:Optional[int]=256,
        return_dict:Optional[bool]=False,
    ) -> ZebraOutput:
        """
        Run the Zebra pipeline to generate knowledge and answer questions.

        Parameters:
        - questions (Union[str, List[str]): 
            The question or list of questions to answer. Can be a single question (str) or a list of questions (List[str]).
        - choices (Optional[Union[List[str], List[List[str]], List[Dict[str, str]], List[List[Dict[str, str]]]]):
            The choices for the questions. 
            If the 'questions' is a string (single question), it can be either a list of strings or a list of dictionaries with the keys 'label' and 'text'.
            example: 
                - ["car", "bus", "train"]
                - [{"label": "A", "text": "car"}, {"label": "B", "text": "bus"}, {"label": "C", "text": "train"}]
            If the 'questions' is a list of strings (question), it can be a either a list of lists of strings or a list of lists of dictionaries.
            example:
                - [["car", "bus", "train"], ["apple", "banana", "orange"]]
                - [[{"label": "A", "text": "car"}, {"label": "B", "text": "bus"}, {"label": "C", "text": "train"}],
                [{"label": "A", "text": "apple"}, {"label": "B", "text": "banana"}, {"label": "C", "text": "orange"}]]
        - top_k (Optional[int]): 
            The number of examples to retrieve.
        - batch_size (Optional[int]): 
            The batch size to use for the retriever.
        - num_kg_examples (Optional[int]): 
            The number of examples to use for the knowledge generation step.
        - num_qa_examples (Optional[int]):
            The number of examples to use for the question answering step.
        - dataset_tag (Optional[str]): 
            The dataset tag used to fetch the relative prompt.
            If you wish to use Zebra for a specific MCQA datasets, the supported options are ["csqa", "obqa", "arc", "piqa", "qasc", "csqa2", "wg"].
            If you wish to use Zebra for a custom dataset, you can specify the dataset tag as "custom".
        - questions_domain (Optional[str]):
            The domain of the questions. The supported options are ["commonsense", "science", "social", "physical"].
        - kg_template_name (Optional[str]):
            The template name to use for the knowledge generation step.
        - qa_template_name (Optional[str]):
            The template name to use for the question answering step.
        - max_generated_knowledge (Optional[int]):
            The maximum number of generated knowledge statemets to use for the question answering step.
        - max_new_tokens (Optional[int]):
            The maximum number of new tokens to generate during the knowledge generation steps.
        - return_dict (Optional[bool]):
            Whether to return the output as a dictionary containing additional information or as a tuple with list of knowledge and answers.

        Returns:
        - ZebraOutput: 
            A dictionary containing the generated knowledge and the answers. 
            If 'return_dict' is True, the output is a dictionary containing: 
                - "knowledge": The generated knowledge.
                - "answers": The answers.
                - "retriever_output": The output of the retriever.
                - "kg_shots": The knowledge generation examples.
                - "samples": The input samples.
        """
        self.log("Running ZEBRA!")
        self.log("Retrieving examples for the input questions.")

        # Retrieve examples for the knowledge generation step.
        retriever_input = self.create_retriever_input(questions, choices)
        retriever_output = self.retrieve(
            query=retriever_input, 
            top_k=top_k, 
            batch_size=batch_size
        )

        if num_kg_examples > top_k:
            self.log(f"""
                The num_kg_examples={num_kg_examples} is greater than the number of retrieved examples top_k={top_k}.
                Setting the number of knowledge generation examples to {top_k}.
                """, 
                type="warning"
            )
            num_kg_examples = top_k

        # Organize retrieved examples for the knowledge generation step.
        kg_shots = create_knowledge_generation_examples(
            retriever_output=retriever_output,
            num_kg_examples=num_kg_examples,
            question_to_explanations=self.question_to_explanations
        )
        # Prepare the questions and choices for the prompt.
        samples = self.create_input_sample(questions, choices)

        all_questions = []
        all_choices = []
        all_knowledge = []
        all_answers_with_knowledge = []

        self.log(f"Running knowledge generation and informed reasoning for the input questions.")
        for sample, retrieved_examples in tqdm(zip(samples, kg_shots), total=len(samples)):

            if "custom" not in dataset_tag and dataset_tag not in ["csqa", "obqa", "arc", "piqa", "qasc", "csqa2", "wg"]:
                raise ValueError("The dataset_tag must be one of ['csqa', 'obqa', 'arc', 'piqa', 'qasc', 'csqa2', 'wg'] or 'custom'.")
            if questions_domain not in ["commonsense", "science", "social", "physical"]:
                raise ValueError("The questions_domain must be one of ['commonsense', 'science', 'social', 'physical']")
            
            # Prepare the prompt for knowledge generation
            prompt_for_knowledge_gen = prepare_sample_for_knowledge_generation(
                sample=sample,
                shot_samples=retrieved_examples,
                dataset_tag=dataset_tag,
                template_name=kg_template_name
            )

            # Get the generated knowledge from the language model.
            generated_knowledge = get_model_knowledge(
                prompt=prompt_for_knowledge_gen,
                model=self.model,
                tokenizer=self.tokenizer,
                max_generated_knowledge=max_generated_knowledge,
                max_new_tokens=max_new_tokens,
                device=self.device
            )
            
            # Prepare the prompt for the question answering task.
            prompt_with_knowledge = prepare_sample_for_mcq(
                sample=sample,
                sample_knowledge=generated_knowledge,
                dataset_tag=dataset_tag,
                questions_domain=questions_domain,
                template_name=qa_template_name,
            )

            # Get the possible labels.
            labels = [choice["label"] for choice in sample["question"]["choices"]]
            # Get the answer from the model.
            answer_with_knowledge = get_model_answer(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_with_knowledge,
                labels=labels,
                max_new_tokens=1,
                return_scores=False
            )

            # Find the index of the matching label
            index = np.argmax(np.array(labels) == answer_with_knowledge)
            choices_text = [choice["text"] for choice in sample["question"]["choices"]]

            answer_with_knowledge = "{}. {}".format(
                answer_with_knowledge,
                choices_text[index]
            )

            all_questions.append(sample["question"]["stem"])
            all_choices.append(sample["question"]["choices"])
            all_knowledge.append(generated_knowledge)
            all_answers_with_knowledge.append(answer_with_knowledge)
        
        self.log("ZEBRA run completed successfully!", type="success")

        if not return_dict:
            return ZebraOutput(
                explanations=all_knowledge,
                answers=all_answers_with_knowledge,
            )
        return ZebraOutput(
            questions=all_questions,
            choices=all_choices,
            explanations=all_knowledge,
            answers=all_answers_with_knowledge,
            retriever_output=retriever_output,
            kg_shots=kg_shots,
            samples=samples
        )
    
    def create_retriever_input(self,
        questions:Union[str, List[str]],
        choices:Union[List[str], List[List[str]], List[Dict[str, str]], List[List[Dict[str, str]]]],
    ) -> List[str]:
        """
        Create the input for the retriever.

        Parameters:
        - questions (Union[str, List[str]): 
            The question or list of questions to answer.
        - choices (Union[List[str], List[List[str], List[Dict[str, str]], List[List[Dict[str, str]]]]):
            The choices for the questions.

        Returns:
        - List[str]: The input for the retriever.
        """
        query = []
        if isinstance(questions, str):
            questions = [questions]
        for current_question, current_choices in zip(questions, choices):
            if isinstance(current_choices[0], str):
                labels = [chr(65 + i) for i in range(len(choices))]
                joined_choices = " [SEP] ".join(
                    [f"{label}. {choice}" for label, choice in zip(labels, current_choices)]
                )
            elif isinstance(current_choices[0], dict):
                if not any("label" in choice for choice in current_choices) or not any("text" in choice for choice in current_choices):
                    raise KeyError("The 'label' or 'text' key is missing from the choices dictionary.")
                joined_choices = " [SEP] ".join(
                    [f"{choice['label']}. {choice['text']}" for choice in current_choices]
                )
            current_query = f"{current_question} [SEP] {joined_choices}"
            query.append(current_query)
        
        return query

    def retrieve(
            self, 
            query:Union[str, List[str]],
            top_k:Optional[int]=5,
            batch_size:Optional[int]=64,
            **kwargs,
        ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant question-knowledge pairs for the input.

        Parameters:
        - query (Union[str, List[str]): 
            The query or list of queries.
        - top_k (Optional[int]):
            The number of examples to retrieve.
        - batch_size (Optional[int]):
            The batch size to use for the retriever.
        - kwargs:
            Additional keyword arguments to pass to the retriever. See GoldenRetriever.retrieve() for more details.

        Returns:
        - List[Dict[str, Any]]: The retrieved examples.
        """
        if not isinstance(query, (str, list)):
            raise ValueError("Parameter 'query' must be a string or a list of strings.")

        retriever_output = self.retriever.retrieve(
            text=query, 
            k=top_k, 
            batch_size=batch_size, 
            kwargs=kwargs
        )

        parsed_retriever_output = self.parse_retriever_output(
            query=query,
            retriever_output=retriever_output
        )

        return parsed_retriever_output
    
    def parse_retriever_output(
            self, 
            query:Union[str, List[str]],
            retriever_output:List[List[RetrievedSample]],
        ) -> List[Dict[str, Any]]:
        """
        Parse the retriever output.

        Parameters:
        - query (Union[str, List[str]): 
            The query or list of queries.
        - retriever_output (List[List[RetrievedSample]]):
            The output of the retriever.
        
        Returns:
        - List[Dict[str, Any]]: The parsed retriever output.
        """
        if isinstance(query, str):
            query = [query]

        parsed_retriever_output = []
        for i, (q, examples) in enumerate(tqdm(
            zip(query, retriever_output), total=len(query)
        )):
            q = {"text": q, "qid": i}
            candidates = [
                {
                    "docid": e.document.id,
                    "score": e.score,
                    "doc": {
                        "id": e.document.id,
                        "contents": e.document.text,
                        "metadata": {
                            "answerKey": e.document.metadata["answerKey"],
                        },
                    },
                }
                for e in examples
            ]
            request = {"query": q, "candidates": candidates}
            parsed_retriever_output.append(request)
        
        return parsed_retriever_output

    def create_input_sample(
        self,
        questions:Union[str, List[str]],
        choices:Union[List[str], List[List[str]], List[Dict[str, str]], List[List[Dict[str, str]]]],
    ):
        """
        Create the input prompt for the question answering task.
        """

        samples = []
        if isinstance(questions, str):
            questions = [questions]
        for current_question, current_choices in zip(questions, choices):
            sample = dict()
            if isinstance(current_choices[0], str):
                sample = {
                    "question": {
                        "stem": current_question,
                        "choices": [
                            {"label": chr(65 + i), "text": choice} for i, choice in enumerate(current_choices)
                        ]
                    }
                }
            elif isinstance(current_choices[0], dict):
                if not any("label" in choice for choice in current_choices) or not any("text" in choice for choice in current_choices):
                    raise ValueError("The 'label' key is missing from the choices dictionary.")
                sample = {
                    "question": {
                        "stem": current_question,
                        "choices": current_choices
                    }
                }
            samples.append(sample)
        return samples