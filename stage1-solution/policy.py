from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel


import env


class Policy(BaseModel, ABC):
    @abstractmethod
    def __call__(self, env: env.Env) -> env.Action: ...

    @classmethod
    def fmt_config(cls, model_dict: dict) -> str:
        # print like this: {policy_class.__name__}(key=value, key=value, ...)
        config_str = ", ".join([f"{key}={value}" for key, value in model_dict.items()])
        return f"{cls.__name__}({config_str})"


class RandomPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> env.Action:
        legal_mask = s.legal_mask()
        p = legal_mask / np.sum(legal_mask)
        return env.Action(np.random.choice(len(p), p=p))

class HumanPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, s: env.State) -> env.Action:
        env.print_state(s)
        print("0 1 2 3 4 5 6")
        legal_mask = s.legal_mask()
        print("legal mask:", legal_mask)
        chosen_action = np.int8(input("Choose action: "))
        return env.Action(chosen_action)