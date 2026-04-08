import os
from openai import OpenAI
from env.environment import EmailEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

env = EmailEnv()

obs = env.reset()

print(f"[START] task=email env=openenv model={MODEL_NAME}")

done = False
step = 0
rewards = []

while not done:

    action = "spam"   # baseline dummy action

    obs, reward, done, info = env.step({
        "action_type": "spam_detection",
        "value": action
    })

    step += 1
    rewards.append(reward)

    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

print(f"[END] success={str(done).lower()} steps={step} rewards={','.join([f'{r:.2f}' for r in rewards])}")