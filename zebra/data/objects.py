from typing import List, Union, Dict, Any
from dataclasses import dataclass

@dataclass
class ZebraOutput:
    """
    Attributes:
        questions (List[str]):
            The list of questions to be answered.
        choices (Union[List[List[str]], List[List[Dict[str, str]]]]):
            The list of choices for each question. Each question can have a list of strings or a list of dictionaries with 'label' and 'text' keys.
        explanations (List[str]):
            The list of explanations generated for each question.
        answers (List[str]):
            The list of answers generated for each question.
        retriever_output (List[Dict[str, Any]]):
            The output from the retriever for each question.
        kg_shots (List[List[Dict[str, Any]]]):
            The knowledge generation examples for each question.
        samples (List[Dict[str, Any]]):
            The input samples created from the questions and choices.
    """
    
    questions: List[str]
    choices: Union[List[List[str]],  List[List[Dict[str, str]]]]
    explanations: List[List[str]]
    answers: List[str]
    retriever_output: List[Dict[str, Any]]
    kg_shots: List[List[Dict[str, Any]]]
    samples: List[Dict[str, Any]]

    def to_dict(self):
        self_dict = {
            "questions": self.questions,
            "choices": self.choices,
            "explanations": self.explanations,
            "answers": self.answers,
            "retriever_output": self.retriever_output,
            "kg_shots": self.kg_shots,
            "samples": self.samples
        }
        return self_dict
