
import argparse
import random
import time
import pickle
import datasets
from gen_utils import get_paper_info, find_influential_citation_info, find_random_reference, is_influential_citation, resume_from_saved_triplets, save_new_triplet

random.seed(42)
start_time = time.time() 

def gen_cited_neg(orig_triplets, start, stride):
    """Generate triples of the form (query, pos, neg) where pos is cited by query and highly influential, and neg is also cited by query but not highly influential"""
    file_name = f"saved_triplets/cited_neg_start_{start}_stride_{stride}.pkl"
    # Continue from where the last result left off
    last_iter, results = resume_from_saved_triplets(file_name)

    for i in range(start if last_iter == -1 else last_iter + stride, len(orig_triplets), stride):
        orig_query = orig_triplets[i]['query']
        q_info = get_paper_info(orig_query['corpus_id'], is_corpus_id = True)

        # Finds a paper tht cited the original query paper and was highly influenced by it to use as the new query paper. The original now becomes the positive paper
        new_q_info = find_influential_citation_info(q_info)
        if not new_q_info:
             continue

        # The neg paper is chosen to be a random paper that the new query paper cited was not highly influential
        neg = find_random_reference(new_q_info, q_info.get('paperId'))
        neg_info = get_paper_info(neg.get('paperId')) if neg else None
        if neg_info and not is_influential_citation(neg_info, new_q_info.get('paperId')):
            if neg_info.get('abstract') not in (None, "None"):
                save_new_triplet(results, new_q_info, q_info, neg_info, orig_query['abstract'], neg_info.get('abstract'))

        with open(file_name, 'wb') as file:
            pickle.dump({'results': results, 'last_iteration': i, 'start': start, 'stride': stride}, file)
        
        if (i + 1) % 300 == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration: {i + 1}, Elapsed Time: {elapsed_time:.2f} seconds, Entries: {len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Triplets where the query paper does cite the neg paper but it is not highly influential')
    parser.add_argument('start', type=int, help='start')
    parser.add_argument('--stride', type=int, default=10, help='Stride value')
    args = parser.parse_args()

    cite_prediction_new = datasets.load_dataset("allenai/scirepeval", "cite_prediction_new")
    gen_cited_neg(cite_prediction_new["train"], args.start, args.stride)
