
import argparse
import random
import time
import pickle
import datasets
import os
from gen_utils import get_paper_info, find_influential_citation, find_random_reference, is_influential_citation, resume_from_saved_triplets

random.seed(42)
start_time = time.time() 

def gen_cited_neg(orig_triplets, start, stride):
    """Generate triples of the form (query, pos, neg) where pos is cited by query and highly influential, and neg is also cited by query but not highly influential"""
    file_name = f"saved_triplets/cited_neg_start_{start}_stride_{stride}.pkl"
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

        new_q_paper_info = get_paper_info(influential_citation.get('paperId'))
        if new_q_paper_info and 'abstract' in new_q_paper_info and any(ref.get('paperId') == q_info.get('paperId') for ref in new_q_paper_info.get('references', [])):
            neg_paper = find_random_reference(new_q_paper_info, q_info.get('paperId'))
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
                                'title': q_info.get('title'),
                                'abstract': q_info.get('abstract'),
                                'corpus_id': q_info.get('corpusId')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Triplets where the query paper does cite the neg paper but it is not highly influential')
    parser.add_argument('start', type=int, help='start')
    parser.add_argument('--stride', type=int, default=10, help='Stride value')
    args = parser.parse_args()

    cite_prediction_new = datasets.load_dataset("allenai/scirepeval", "cite_prediction_new")
    gen_cited_neg(cite_prediction_new["train"], args.start, args.stride)
