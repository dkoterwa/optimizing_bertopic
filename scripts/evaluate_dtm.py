import numpy as np
import os
from BERTopic_evaluation.evaluation import Trainer, DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
import os
from utils import DEFAULT_RANDOM_SEED, seedEverything


def evaluate_dataset(embeddings_path, save_path, number_of_runs, dataset_name):
    embeddings_data = np.load(embeddings_path, allow_pickle=True).item()
    dataset, custom = f"{dataset_name}_dtm", True

    for key in tqdm(embeddings_data.keys()):
        embeddings = np.array(embeddings_data[key])
        for i in range(number_of_runs):
            params = {
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "nr_topics": [50],
                "min_topic_size": 15,
                "verbose": True,
                }

            data_loader = DataLoader("trump_dtm")
            _, timestamps = data_loader.load_docs()
            
            # Match indices
            with open(f"{dataset}/indexes.txt") as f:
                indices = f.readlines()

            indices = [int(index.split("\n")[0]) for index in indices]
            timestamps = [timestamp for index, timestamp in enumerate(timestamps) if index in indices]

            trainer = Trainer(
                dataset=dataset,
                model_name="BERTopic",
                params=params,
                bt_embeddings=embeddings,
                bt_timestamps=timestamps,
                topk=5,
                bt_nr_bins=10,
                custom_dataset=custom,
                verbose=True,
            )
            results = trainer.train(save=f"{save_path}BERTopic_{dataset_name}_{(key)}_{i+1}")

if __name__ == "__main__":
    seedEverything(DEFAULT_RANDOM_SEED)
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings_path",
        type=str,
        help="Path to the .npy file with saved embeddings"
    )
    parser.add_argument(
        "--results_save_path", type=str, help="Path to save our results"
    )
    parser.add_argument(
        "--n_of_runs", type=int, help="How many evaluations to run for single combination", default=3
    )
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the directory where corpus of dataset is stored"
    )
    args = parser.parse_args()
    os.makedirs(args.results_save_path, exist_ok=True)
    evaluate_dataset(args.embeddings_path, args.results_save_path, args.n_of_runs, args.dataset_name)
