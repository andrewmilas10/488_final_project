
import requests
import random
import time
import pickle
import datasets

cite_prediction_new = datasets.load_dataset("allenai/scirepeval", "cite_prediction_new")

def get_paper_info(paper_id, is_corpus_id = False, retry_attempts=3):
    url = f"https://api.semanticscholar.org/v1/paper/{'CorpusId:' if is_corpus_id else ''}{paper_id}"
    headers = {'X-API-KEY': "YwS0Vy4Ix91BcoAYYAk9L2XV1IMyUD3c7nsR8rPe"}

    for attempt in range(retry_attempts):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Gots error {response}\nRetrying in 20 seconds")
            time.sleep(14)

    # Return None if all retry attempts fail
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

random.seed(42)
results = []

print(len(cite_prediction_new["train"]))
start_time = time.time() 

start = 24375
stride = 10 

for i in range(start, len(cite_prediction_new["train"]), stride):
    query_corpus_id = cite_prediction_new["train"][i]['query']['corpus_id']
    query_paper_info = get_paper_info(query_corpus_id, is_corpus_id = True)
    if query_paper_info:
        influential_citation = find_influential_citation(query_paper_info)
        if influential_citation:
            new_q_paper_info = get_paper_info(influential_citation.get('paperId'))
            if new_q_paper_info and 'abstract' in new_q_paper_info and any(ref.get('paperId') == query_paper_info.get('paperId') for ref in new_q_paper_info.get('references', [])):
                neg_paper = find_random_reference(new_q_paper_info, query_paper_info.get('paperId'))
                if neg_paper:
                    neg_paper_info = get_paper_info(neg_paper.get('paperId'))
                    if neg_paper_info and not is_influential_citation(neg_paper_info, new_q_paper_info.get('paperId')):
                        if new_q_paper_info.get('abstract') and neg_paper_info.get('abstract'):
                            results.append({
                                'query': {
                                    'title': new_q_paper_info.get('title'),
                                    'abstract': new_q_paper_info.get('abstract'),
                                    'corpus_id': new_q_paper_info.get('corpusId')
                                },
                                'pos': {
                                    'title': query_paper_info.get('title'),
                                    'abstract': query_paper_info.get('abstract'),
                                    'corpus_id': query_paper_info.get('corpusId')
                                },
                                'neg': {
                                    'title': neg_paper_info.get('title'),
                                    'abstract': neg_paper_info.get('abstract'),
                                    'corpus_id': neg_paper_info.get('corpusId'),
                                    'score': -1
                                }
                            })
                            # Pickling the data
                            with open(f'results_start_{start}_stride_{stride}.pkl', 'wb') as file:
                                pickle.dump({'results': results, 'last_iteration': i, 'start': start, 'stride': stride}, file)
    if (i + 1) % 300 == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration: {i + 1}, Elapsed Time: {elapsed_time:.2f} seconds, Entries: {len(results)}")

# results now contains the required list of objects
print(len(results))
