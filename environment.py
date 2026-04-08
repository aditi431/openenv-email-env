import random
from .dataset import EMAIL_DATA
from .models import Observation

class EmailEnv:

    def __init__(self):

        self.index = 0
        self.data = EMAIL_DATA
        self.steps = 0

    def reset(self):

        self.index = 0
        self.steps = 0

        email = self.data[self.index]

        return Observation(
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"]
        )

    def state(self):

        return {
            "index": self.index,
            "steps": self.steps
        }

    def step(self, action):

        email = self.data[self.index]

        reward = 0
        done = False
        info = {}

        if action["action_type"] == "spam_detection":

            if action["value"] == email.get("label"):
                reward = 1.0

        if action["action_type"] == "priority":

            if action["value"] == email.get("priority"):
                reward = 1.0

        if action["action_type"] == "routing":

            if action["category"] == email.get("category"):
                reward += 0.4

            if action["department"] == email.get("department"):
                reward += 0.6

        self.index += 1
        self.steps += 1

        if self.index >= len(self.data):

            done = True
            observation = None

        else:

            next_email = self.data[self.index]

            observation = Observation(
                email_id=next_email["id"],
                subject=next_email["subject"],
                body=next_email["body"]
            )

        return observation, reward, done, info