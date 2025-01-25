from langchain_core.tools import tool

@tool("chitchat_answer")
def chitchat_answer(
    chitchat_answer: str,
):
    """"Use this tool to return a natural language response to the user if the query is not about a news
    in general in the form of a plain and simple response especially if it falls under a chitchat utterance.
    """
    return chitchat_answer