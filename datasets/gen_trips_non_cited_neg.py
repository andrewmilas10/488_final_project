import argparse
import random
import time
import pickle
import datasets
from gen_utils import get_paper_info, find_influential_citation_info, resume_from_saved_triplets, save_new_triplet

random.seed(42)
start_time = time.time() 

def gen_non_cited_neg(orig_triplets, start, stride, hard):
    """Generate triples of the form (query, pos, neg) where pos is cited by query and highly influential, and neg is not cited by query. 
    If hard is true, then neg will be cited by pos. Otherwise, neg will not be cited by pos"""
    file_name = f"saved_triplets/non_cited_neg_start_{start}_stride_{stride}_hard_{hard}.pkl"
    last_iter, results = resume_from_saved_triplets(file_name) # Continue from where the last result left off

    for i in range(start if last_iter == -1 else last_iter + stride, len(orig_triplets), stride):
        orig_query = orig_triplets[i]['query']
        q_info = get_paper_info(orig_query['corpus_id'], is_corpus_id = True)

        # Finds a paper tht cited the original query paper and was highly influenced by it to use as the new query paper. The original now becomes the positive paper
        new_q_info = find_influential_citation_info(q_info)
        if not new_q_info:
             continue

        # The neg paper is chosen to be one either cited or not cited by the 'original query'/'new positive' paper based if hard negatives are desired
        neg = orig_triplets[i]['pos' if hard else'neg']
        neg_info = get_paper_info(neg['corpus_id'], is_corpus_id=True)

        # Ensures the neg paper is truly not cited as a reference of the new query paper and if so saves the triplet
        if neg_info and not any(ref.get('paperId') == neg_info.get('paperId') for ref in new_q_info.get('references', [])):
            save_new_triplet(results, new_q_info, q_info, neg_info, orig_query['abstract'], neg['abstract'])
    
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
