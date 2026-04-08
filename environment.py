from pydantic import BaseModel
from typing import Tuple, Dict
from .dataset import EMAIL_DATA


class Observation(BaseModel):
    email: str
    index: int


class EmailEnv:

    def __init__(self):
        self.index = 0
        self.done = False

    def reset(self) -> Observation:
        self.index = 0
        self.done = False
        return Observation(
            email=EMAIL_DATA[self.index]["text"],
            index=self.index
        )

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict]:

        correct_label = EMAIL_DATA[self.index]["label"]

        reward = 1.0 if action == correct_label else -0.2

        self.index += 1

        if self.index >= len(EMAIL_DATA):
            self.done = True
            obs = Observation(email="", index=self.index)
        else:
            obs = Observation(
                email=EMAIL_DATA[self.index]["text"],
                index=self.index
            )

        return obs, reward, self.done, {}

    def state(self):
        return {
            "index": self.index,
            "done": self.done
        }
