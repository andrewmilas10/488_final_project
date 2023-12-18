import requests
import time
import os
import random
import pickle
from dotenv import load_dotenv

load_dotenv()

RETRY_ATTEMPTS = 3
TIMEOUT = 40
API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')

random.seed(42)

def get_paper_info(paper_id, is_corpus_id=False):
    """Queries the semantic scholar API for information about a given paper. The paper_id parameter can be set as a
    semantic scholar corpus_id when is_corpus_id is True. Otherwise, it is interpreted as a regular semantic scholar paper_id"""
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
    """Returns a new paper that cites the one passed in and was highly influenced by it's contents"""
    influential_citations = [citation for citation in paper_info.get('citations', []) if citation.get('isInfluential', False)]
    if influential_citations:
        return random.choice(influential_citations)
    return None

def find_random_reference(paper_info, exclude_id):
    """Pick out a random paper that the passed in paper cites"""
    references = [reference for reference in paper_info.get('references', []) if reference.get('paperId') != exclude_id]
    if references:
        return random.choice(references)
    return None

def is_influential_citation(source_paper_info, target_paper_id):
    """Checks if the target paper cited the source paper and if the citation was highly influential"""
    for citation in source_paper_info.get('citations', []):
        if citation.get('paperId') == target_paper_id and citation.get('isInfluential', False):
            return True
    return False

def resume_from_saved_triplets(file_name):
    """Returns the list of triplets and the last iteration a triplet was found for saved pickle file of results.
    These are returned as tuple of the form (last_iter, triplets) with last_iter being -1 if no triplets exist"""
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
            results = data['results']
            last_iter = data['last_iteration']
            print(f"Already have {len(results)} triples. Starting at idx {last_iter}")
            return last_iter, results
    print -1, []
