import argparse
import random
import time
import pickle
import datasets
from utils import get_paper_info, find_influential_citation

random.seed(42)
start_time = time.time() 

def generate_queries(orig_triplets, start, stride, hard):
    results = []
    next_iter = start
    # Continue from where the last result left off
    with open(f"new_queries_start_{old_start}_stride_{stride}_hard_{hard}.pkl", 'rb') as file:
        data = pickle.load(file)
        results = data['results']
        next_iter = data['last_iteration']+stride

    for i in range(next_iter, len(orig_triplets), stride):
        q_id = orig_triplets[i]['query']['corpus_id']
        q_info = get_paper_info(q_id, is_corpus_id = True)
        if q_info:
            influential_citation = find_influential_citation(q_info)
            if influential_citation:
                new_q_info = get_paper_info(influential_citation.get('paperId'))
                abstract = new_q_info.get('abstract') if new_q_info else None
                if new_q_info and abstract != "None" and abstract is not None and any(ref.get('paperId') == q_info.get('paperId') for ref in new_q_info.get('references', [])):
                    neg = cite_prediction_new["train"][i]['pos']['corpus_id'] if hard else \
                        cite_prediction_new["train"][i]['neg']['corpus_id'] 
                    neg_paper_info = get_paper_info(neg, is_corpus_id=True)
                    if neg_paper_info and not any(ref.get('paperId') == neg_paper_info.get('paperId') for ref in new_q_info.get('references', [])):
                        results.append({
                            'query': {
                                'title': new_q_info.get('title'),
                                'abstract': new_q_info.get('abstract'),
                                'corpus_id': new_q_info.get('corpusId')
                            },
                            'pos': {
                                'title': q_info.get('title'),
                                'abstract': cite_prediction_new["train"][i]['query']['abstract'],
                                'corpus_id': q_info.get('corpusId')
                            },
                            'neg': {
                                'title': neg_paper_info.get('title'),
                                'abstract': cite_prediction_new["train"][i]['pos' if hard else 'neg']['abstract'],
                                'corpus_id': neg_paper_info.get('corpusId'),
                                'score': -1
                            }
                        })
                    with open(f'new_queries_start_{old_start}_stride_{stride}_hard_{hard}.pkl', 'wb') as file:
                        pickle.dump({'results': results, 'last_iteration': i, 'start': start, 'stride': stride, 'hard': hard}, file)
        if (i + 1) % 1000 == start+1:
            elapsed_time = time.time() - start_time
            print(f"Iteration: {i + 1}, Elapsed Time: {elapsed_time:.2f} seconds, Entries: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Query Script')
    parser.add_argument('start', type=int, help='start')
    parser.add_argument('--stride', type=int, default=10, help='Stride value')
    args = parser.parse_args()
    hard_neg = args.old_start >= 7

    cite_prediction_new = datasets.load_dataset("allenai/scirepeval", "cite_prediction_new")
    generate_queries(cite_prediction_new["train"], args.old_start, args.stride, hard)



old_start = 3
stride = 10 
hard = old_start >= 7
with open(f"new_queries_start_{old_start}_stride_10_hard_{hard}.pkl", 'rb') as file:
    # Load and deserialize the data from the file
    data = pickle.load(file)

    # Extracting the results and last iteration
    results = data['results']
    start = data['last_iteration']+stride

    # Print the retrieved data
    print(f"Next Iteration: {start}")
    print(f"Total Entries: {len(results)}")
print(len(results))



# results now contains the required list of objects
print(len(results))
