import requests
import time
import os
import random
from dotenv import load_dotenv

load_dotenv()

RETRY_ATTEMPTS = 3
TIMEOUT = 40
API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')

random.seed(42)

def get_paper_info(paper_id, is_corpus_id=False):
    url = f"https://api.semanticscholar.org/v1/paper/{'CorpusId:' if is_corpus_id else ''}{paper_id}"
    for _ in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, headers={'X-API-KEY': API_KEY}) if API_KEY else requests.get(url)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"An exception occurred: {e}\nRetrying in {TIMEOUT} seconds")

        time.sleep(TIMEOUT) 

    return None

def find_influential_citation(paper_info):
    influential_citations = [citation for citation in paper_info.get('citations', []) if citation.get('isInfluential', False)]
    if influential_citations:
        return random.choice(influential_citations)
    return None

def find_random_reference(paper_info, exclude_id):
    references = [reference for reference in paper_info.get('references', []) if reference.get('paperId') != exclude_id]
    if references:
        return random.choice(references)
    return None

def is_influential_citation(source_paper_info, target_paper_id):
    for citation in source_paper_info.get('citations', []):
        if citation.get('paperId') == target_paper_id and citation.get('isInfluential', False):
            return True
    return False