import os
from semantic_router import Route
from semantic_router import RouteLayer
from semantic_router.encoders import OpenAIEncoder
from datetime import datetime
import pytz

os.getenv("OPENAI_API_KEY")


time_route = Route(
    name="get_current_date",
    utterances=[
        "what is the current news?",
        "what is the latest news?",
        "Whats the most tragic thing happened last month?",
        "Last years news about Covid19"
    ],
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
        "Hi",
        "How are you?",
        "That's great thanks!"
    ],
)

routes = [time_route, chitchat]
# encoder = OpenAIEncoder(name="text-embedding-3-small")

rl = RouteLayer(encoder=OpenAIEncoder(name="text-embedding-3-small"), routes=routes)


def get_current_date():
    timezone = pytz.timezone('Asia/Manila')  # Replace with your desired time zone
    now = datetime.now(timezone)
    human_readable_date = now.strftime('%B %d, %Y')
    return (
        f"The current date is {human_readable_date}, use "
        "this information in your response"
    )

def semantic_layer(query: str):
    route = rl(query)
    if route.name == "get_current_date":
        query += f" (SYSTEM NOTE: {get_current_date()})"
    elif route.name == "chitchat":
        query += f" (SYSTEM NOTE: A type of chitchat utterance)"
    else:
        pass
    return query