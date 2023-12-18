import argparse
import random
import time
import pickle
import datasets
import os
from gen_utils import get_paper_info, find_influential_citation, resume_from_saved_triplets

random.seed(42)
start_time = time.time() 

def gen_non_cited_neg(orig_triplets, start, stride, hard):
    """Generate triples of the form (query, pos, neg) where pos is cited by query and highly influential, and neg is not cited by query. 
    If hard is true, then neg will be cited by pos. Otherwise, neg will not be cited by pos"""
    file_name = f"saved_triplets/non_cited_neg_start_{start}_stride_{stride}_hard_{hard}.pkl"
    # Continue from where the last result left off
    last_iter, results = resume_from_saved_triplets(file_name)

    for i in range(start if last_iter == -1 else last_iter + stride, len(orig_triplets), stride):
        q_id = orig_triplets[i]['query']['corpus_id']
        q_info = get_paper_info(q_id, is_corpus_id = True)
        if not q_info:
            continue
        influential_citation = find_influential_citation(q_info)
        if not influential_citation:
            continue

        # The influential citation is now the new query paper since we know it cited the original query paper. The original now becomes the positive paper
        new_q_info = get_paper_info(influential_citation.get('paperId'))
        abstract = new_q_info.get('abstract') if new_q_info else None
        if new_q_info and abstract != "None" and abstract is not None and any(ref.get('paperId') == q_info.get('paperId') for ref in new_q_info.get('references', [])):
            # The neg paper is chosen to be one either cited or not cited by original query paper based if hard negatives are desired
            neg = orig_triplets[i]['pos' if hard else'neg']['corpus_id']

            neg_info = get_paper_info(neg, is_corpus_id=True)
            if neg_info and not any(ref.get('paperId') == neg_info.get('paperId') for ref in new_q_info.get('references', [])):
                results.append({
                    'query': {
                        'title': new_q_info.get('title'),
                        'abstract': new_q_info.get('abstract'),
                        'corpus_id': new_q_info.get('corpusId')
                    },
                    'pos': {
                        'title': q_info.get('title'),
                        'abstract': orig_triplets[i]['query']['abstract'],
                        'corpus_id': q_info.get('corpusId')
                    },
                    'neg': {
                        'title': neg_info.get('title'),
                        'abstract': orig_triplets[i]['pos' if hard else 'neg']['abstract'],
                        'corpus_id': neg_info.get('corpusId'),
                        'score': -1
                    }
                })
            with open(file_name, 'wb') as file:
                    pickle.dump({'results': results, 'last_iteration': i, 'start': start, 'stride': stride, 'hard': hard}, file)
                    
        if (i + 1) % 1000 == start+1:
            elapsed_time = time.time() - start_time
            print(f"Iteration: {i + 1}, Elapsed Time: {elapsed_time:.2f} seconds, Entries: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Triplets where the query paper does not cite the neg paper')
    parser.add_argument('start', type=int, help='start')
    parser.add_argument('--stride', type=int, default=10, help='Stride value')
    args = parser.parse_args()
    hard_neg = args.start >= 7

    cite_prediction_new = datasets.load_dataset("allenai/scirepeval", "cite_prediction_new")
    gen_non_cited_neg(cite_prediction_new["train"], args.start, args.stride, hard_neg)
