from typing import Any, Dict, List


def load_evaluation_data(data_path: str, limit_samples: int) -> List[Dict[str, Any]]:
    """
    Load the evaluation data from the given path.
    """
    with open(data_path, "r") as f_in:
        eval_data = f_in.readlines()
        if limit_samples:
            eval_data = eval_data[:limit_samples]
    return eval_data
