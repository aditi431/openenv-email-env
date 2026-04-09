from .dataset import EMAIL_DATA
from .models import Observation

class EmailEnv:

    def __init__(self):
        self.index = 0
        self.done = False

    def reset(self):
        self.index = 0
        self.done = False
        return Observation(
            email=EMAIL_DATA[self.index]["text"],
            index=self.index
        )

    def step(self, action):
        correct = EMAIL_DATA[self.index]["label"]

        reward = 1.0 if action == correct else -0.2

        self.index += 1

        if self.index >= len(EMAIL_DATA):
            self.done = True
            return None, reward, True, {}

        obs = Observation(
            email=EMAIL_DATA[self.index]["text"],
            index=self.index
        )

        return obs, reward, False, {}

    def state(self):
        return {"index": self.index}
