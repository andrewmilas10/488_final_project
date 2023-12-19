# 488_final_project

## Preparing Datasets

In order to generate the various types of triples used for our datasets, simply download dependencies from our requirements.txt file run the start_gen_cited_neg.sh and start_gen_non_cited_neg.sh bash scripts in the datasets directory. Note, an environment variable SEMANTIC_SCHOLAR_API_KEY can be set to reduce rate limits imposed by the Semantic Scholar API, though the scripts should still work without them. 

```bash
git clone https://github.com/andrewmilas10/488_final_project.git
cd datasets
pip install -r requirements.txt
bash ./start_gen_cited_neg.sh
bash ./start_gen_non_cited_neg.sh
```

start_gen_cited_neg.sh will generate 'hard HIC triplets' where negative papers are cited by the query paper but not influential and start_gen_non_cited_neg.sh will generate 'easy/medium HIC triplets' where the negative paper isn't cited by the query paper and where medium HIC triplets are cited by a paper cited by the query paper. These scripts will spawn processes in the background to collect triplets and save them in the datasets/saved_triplets directory with logs shown in datasets/logs. The processes can be killed altogether using "pkill -f"

Finally, after triplets are generated, the steps in the datasets/publish_datasets_to_hf.ipynb notebook can be followed to generate the datasets by mixing different sets of triplets and push them to HuggingFace. The HF_ACCESS_TOKEN and HF_USERNAME environment variables should be appropriately set for this to work. 

### Hosted Datasets

The datasets generated and used in our class final report are hosted on HuggingFace:

- [Baseline](https://huggingface.co/datasets/cheafdevo56/NoInfluentials)
- [Dataset A (Influential_CitedNegs_1_Percent)](https://huggingface.co/datasets/cheafdevo56/Influential_CitedNegs_1_Percent)
- [Dataset B (Influential_CitedNegs_5_Percent)](https://huggingface.co/datasets/cheafdevo56/Influential_CitedNegs_5_Percent)
- [Dataset C (Influential_CitedNegs_10_Percent)](https://huggingface.co/datasets/cheafdevo56/Influential_CitedNegs_10_Percent)
- [Dataset D (Influential_NonCitedNegs_10_Percent)](https://huggingface.co/datasets/cheafdevo56/Influential_NonCitedNegs_10_Percent)
- [Dataset E (Influential_MixedNegTypes_10_Percent)](https://huggingface.co/datasets/cheafdevo56/Influential_MixedNegTypes_10_Percent)
- [Similar to Dataset D but 100,000 samples (Influential_NonCitedNegs_10_Percent_large)](https://huggingface.co/datasets/cheafdevo56/Influential_NonCitedNegs_10_Percent_large)

We also share datasets of only our new types of triples at:

- [All Hard HIC Triplets](https://huggingface.co/datasets/cheafdevo56/All_Hard_HIC_Triplets)
- [All EASY MEDIUM HIC Triplets](https://huggingface.co/datasets/cheafdevo56/All_EASY_MEDIUM_HIC_Triplets)


### Fine-tuning and Evaluation

We fine-tuned SPECTER2 and evaluated our models on Google Colab. We share the python notebook used to do this at evaluation/Running_Specter.ipynb, which can be followed to reproduce the steps we took for fine-tuning and evaluation. Note, we included a forked version of the [scirepeval](https://github.com/allenai/scirepeval) project, which we modified slightly to update dependencies needed. Ensure the dependencies from 'evaluation/scirepeval/requirements.txt' are used for fine-tuning and evaluation. Note, in the notebook many of the uninstalls we found were necessary for use in Colab's execution environment, but perhaps could be skipped in a virtual environment of a non-Colab setup. 

### Hosted Models

The models generated and used in our class final report are hosted on HuggingFace:

- [Baseline](https://huggingface.co/deyuanli/no_influentials_baseline)
- [Model A](https://huggingface.co/deyuanli/specter_1_percent)
- [Model B](https://huggingface.co/deyuanli/specter_5_percent)
- [Model C](https://huggingface.co/deyuanli/specter_10_percent)
- [Model D](https://huggingface.co/deyuanli/specter_noncited_negs_10)
- [Model E](https://huggingface.co/deyuanli/specter_mixed_5_5)
- [Model D Large](https://huggingface.co/deyuanli/large_10)