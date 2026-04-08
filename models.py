from pydantic import BaseModel

class Observation(BaseModel):
    email_id: int
    subject: str
    body: str

class Action(BaseModel):
    action_type: str
    value: str

class Reward(BaseModel):
    score: float