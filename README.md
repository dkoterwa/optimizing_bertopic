# Optimizing Topic Modeling with BERTopic: Exploring Multiple Embedding Configurations and Stop Words Influence

This is a repository with scripts needed to reproduce work presented in the paper.

## Getting results
To reproduce the research and obtain results of BERTopic evaluation, follow the steps below.

**1. Install requirements**
```bash
pip install -r requirements.txt
```

**2. Get sentence embeddings**
```bash
python retrieve_embeddings.py --corpus_path <path to the tsv files with texts> --embedding_model_name <Hugging Face name of the model> --dataset_name <name for saving purpose>
```

**3. Run evaluation**
```bash
python evaluate.py --embeddings_path <path to the .npy file with saved embeddings> --results_save_path <path to save results> --n_of_runs <number of runs for single combination> --dataset_name <name of the directory where .tsv file with corpus is stored>
```

After completing these steps, json files with results will be generated and saved in specified directory. Later you can follow the steps I've done in notebooks to visualize the results.

