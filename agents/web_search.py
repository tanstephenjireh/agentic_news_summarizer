import streamlit as st
from serpapi import GoogleSearch
from langchain_core.tools import tool
from dotenv import dotenv_values

config = dotenv_values(".env")
serpapi = config["SERPAPI_KEY"]


serpapi_params = {
    "engine": "google",
    "api_key": serpapi
}

@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5
    })

    # st.write(search.get_dict().keys())
    # print(search.get_dict().keys())

    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts