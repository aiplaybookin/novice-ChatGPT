from GoogleNews import GoogleNews
import spacy
import pandas as pd
from newspaper import Article
import nltk
from llama_index import Document, GPTSimpleVectorIndex
import requests

# Load the spacy model
nlp = spacy.load("en_core_web_lg")

# TODO : Remove and use LLMs to classify titles and score
def scoreNewsTitle(title, keyword):
    """
    Compute score given keyword is related to the news title using word embeddings.

    Args:
    title (str): The news title to check.
    keyword (str): The keyword to search for.

    Returns:
    float: 0 to 1.
    """
    # Process the title and keyword using spacy
    title_doc = nlp(title)
    keyword_doc = nlp(keyword)

    # Compute the similarity between the title and keyword
    similarity = title_doc.similarity(keyword_doc)

    return similarity


# Utilities
def getArticleText(url):
  article = Article(url)
  article.download()
  article.parse()
  return article.text

def getSummary(articleText):
  t = Document(articleText)
  index = GPTSimpleVectorIndex.from_documents([t])
  summary = index.query("Summarise the news story. Output text should only contain summary with no references to document.")
  return str(summary)

# Get original URL
def get_final_url(url):
    url = "https://" + url
    try:
        response = requests.get(url, allow_redirects=False)
        final_url = response.headers['Location']
        return final_url
    except requests.exceptions.RequestException as e:
        print("Error: ", e)
        return None


def fetch_news(newsKeyword):
    googlenews = GoogleNews()
    googlenews = GoogleNews(lang='en', region='INDIA', period='1d')
    googlenews.get_news(newsKeyword)
    newsResults = googlenews.results()
    newsTitles = googlenews.get_texts()

    # Score the News Title for Keyword

    score =[]
    for newsTitle in newsTitles:
        result = scoreNewsTitle(newsTitle, newsKeyword)
        score.append(result)

    titleDf = pd.DataFrame({'title' : newsTitles, 'score' : score})
    titleDf.sort_values(by='score', ascending=False, inplace=True)



    # Get Summary and Classification score on topic
    newsSummary=[]
    newsOriginalLinks=[]
    newsLinks = googlenews.get_links()

    # Getting first 20 link only as of now 
    for newsLink in newsLinks[:10] :
        
        url = 'https://'+ newsLink
        
        # Get original link
        newsOriginalLinks.append(get_final_url(newsLink))

        # Get article text
        atext = getArticleText(url)

        # Get summary
        newsSummary.append(getSummary(atext))

    # Collate news items
    news_items = []
    for i in range(len(newsSummary)):
        news_item = {'title': newsTitles[i], 'url': newsOriginalLinks[i], 'summary': newsSummary[i]}
        news_items.append(news_item)

    return news_items
