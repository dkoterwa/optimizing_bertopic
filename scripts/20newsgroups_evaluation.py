import numpy as np
from bertopic_evaluation import Trainer
from tqdm import tqdm
from argparse import ArgumentParser
import os
from utils.utils import DEFAULT_RANDOM_SEED, seedEverything


def evaluate_20ng_dataset(embeddings_path, save_path, number_of_runs):
    embeddings_data = np.load(embeddings_path, allow_pickle=True).item()
    dataset, custom = "./data/20newsgroups", True
    for key in tqdm(embeddings_data.keys()):
        embeddings = np.array(embeddings_data[key])
        for i in range(number_of_runs):
            params = {
                "embedding_model": "all-MiniLM-L12-v2",
                "nr_topics": [(i + 1) * 10 for i in range(5)],
                "min_topic_size": 15,
                "verbose": True,
            }
            trainer = Trainer(
                dataset=dataset,
                model_name="BERTopic",
                params=params,
                bt_embeddings=embeddings,
                custom_dataset=custom,
                verbose=True,
            )
            results = trainer.train(save=f"{save_path}BERTopic_20ng_{(key)}_{i+1}")


if __name__ == "__main__":
    seedEverything(DEFAULT_RANDOM_SEED)
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings_path",
        type=str,
        help="Path to the .npy file with saved embeddings",
        default="./embeddings_data/embeddings_20ng.npy",
    )
    parser.add_argument(
        "--results_save_path", type=str, help="Path to save our results", default="./results/20newsgroups/"
    )
    parser.add_argument(
        "--n_of_runs", type=int, help="How many evaluations to run for single combination", default=3
    )
    args = parser.parse_args()
    os.makedirs(args.results_save_path, exist_ok=True)
    evaluate_20ng_dataset(args.embeddings_path, args.results_save_path, args.n_of_runs)
