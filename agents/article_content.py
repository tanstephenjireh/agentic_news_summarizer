from newspaper import Article
from langchain_core.tools import tool

@tool("article_content")
def article_content(article_url: str):
    """Returns the article content from a URL. Can be used directly
    if there's already a URL that needs to be looked at. Also consider
    using this multiple times to look at a certain same topic of article if needed."""
    article = Article(article_url)
    article.download()
    article.parse()

    return article.text