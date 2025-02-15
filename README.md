## Reproducing the work

### Installing requirements
```pip install -r requirements.txt```

### Preparing embeddings
To create embeddings, move to the `/scripts` directory and run `retrieve_embeddings.py`. <br><br> Example run:

```python
python retrieve_embeddings.py \
    --corpus_path ..data/with_stopwords/trump/corpus.tsv \
    --dataset_name trump
```

### Evaluation
After preparing the embeddings, you can run BERTopic evaluation to assess whether any embedding configuration achieves better results. To run evaluation, move to the `/scripts` directory and run `evaluate.py` in case of traditional topic modeling or `evaluate_dtm.py` if you are working with dynamic topic modeling. <br><br> Example run:

```python
python evaluate_bertopic.py \
    --embeddings_path ../embeddings_data/with_stopwords/trump/trump_with_stopwords.npy \
    --results_save_path ../evaluation_results/ \
    --dataset_name sample_dataset \
    --has_stopwords
```


