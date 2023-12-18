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

def find_influential_citation_info(paper_info):
    """Returns semantic scholar information about a new paper that cites the one passed in and was highly influenced by it's contents. 
    Ensures the new paper has an abstract. Returns None if such a highly influential paper wasn't found"""
    if not paper_info:
        return None

    influential_citations = [c for c in paper_info.get('citations', []) if c.get('isInfluential')] 
    if influential_citations:
        citation =  random.choice(influential_citations)
        c_info = get_paper_info(citation.get('paperId'))
        # Ensures the citation has an abstract and actually has the given paper as a reference
        if c_info and c_info.get('abstract') not in (None, "None") and \
            any(ref.get('paperId') == paper_info.get('paperId') for ref in c_info.get('references', [])): 
            return c_info
    return None

def find_random_reference(paper_info, exclude_id):
    """Pick out a random paper that the passed in paper cites. Picks one whose paperId isn't exclude_id"""
    references = [ref for ref in paper_info.get('references', []) if ref.get('paperId') != exclude_id]
    if references:
        return random.choice(references)
    return None

def is_influential_citation(source_paper_info, target_paper_id):
    """Checks if the target paper cited the source paper and if the citation was highly influential"""
    for citation in source_paper_info.get('citations', []):
        if citation.get('paperId') == target_paper_id and citation.get('isInfluential'):
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
    return -1, []

def save_new_triplet(results, q_info, pos_info, neg_info, pos_abstact, neg_abstract):
    """Creates a triplet in the format expected for finetuning with scirepeval. For the positive and neg papers, abstracts are passed
    in separately from semantic scholar info for the papers. However, for query paper, it must be passed in all together"""
    results.append({
        'query': {
            'title': q_info.get('title'),
            'abstract': q_info.get('abstract'),
            'corpus_id': q_info.get('corpusId')
        },
        'pos': {
            'title': pos_info.get('title'),
            'abstract': pos_abstact,
            'corpus_id': pos_info.get('corpusId')
        },
        'neg': {
            'title': neg_info.get('title'),
            'abstract': neg_abstract,
            'corpus_id': neg_info.get('corpusId'),
            'score': -1
        }
    })
