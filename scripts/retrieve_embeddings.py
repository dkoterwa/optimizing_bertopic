from transformers import AutoTokenizer, AutoModel
import pickle
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import gc
from argparse import ArgumentParser
from utils import StringDataset, DEVICE


class Pooling:
    def __init__(self, pooling_type):
        self.pooling_type = pooling_type

    def __call__(self, hidden_states, layer_number, attention_mask=None):
        if self.pooling_type == "cls":
            return hidden_states[layer_number][:, 0, :]
        elif self.pooling_type == "mean":
            token_embeddings = hidden_states[
                layer_number
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif self.pooling_type == "max":
            token_embeddings = hidden_states[layer_number]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[input_mask_expanded == 0] = (
                -1e9
            )  # Set padding tokens to large negative value
            max_embeddings = torch.max(token_embeddings, 1)[0]
            return max_embeddings
        else:
            raise ValueError("Wrong pooling method provided in the Pooler initialization")


class EmbeddingsRetriever:
    def __init__(self, embedding_model, tokenizer, dataloader):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.functions = [
            self.get_embedding_layer_output,
            self.get_embedding_last_hidden_layer,
            self.get_embedding_sum_all_layers,
            self.get_embedding_second_last_layer,
            self.get_embedding_sum_last_four_layers,
            self.get_embedding_concat_last_four_layers,
        ]

    def tokenize_and_produce_model_output(self, data):
        encoded_input = self.tokenizer(
            data, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        self.attention_mask = encoded_input["attention_mask"].to(DEVICE)
        input_ids = encoded_input["input_ids"].to(DEVICE)
        with torch.no_grad():
            self.model_output = self.embedding_model(
                input_ids=input_ids, attention_mask=self.attention_mask, output_hidden_states=True
            )

    def get_embedding_from_layer(self, layer_number, pooling="cls"):
        hidden_states = self.model_output["hidden_states"]
        # print(f"Hidden states shape {hidden_states[0].shape}")
        if pooling == "cls":
            pooler = Pooling("cls")
            return pooler(hidden_states, layer_number)

        elif pooling == "mean":
            assert (
                self.attention_mask != None
            ), "Please provide attention mask if you are using mean pooling"
            pooler = Pooling("mean")
            return pooler(hidden_states, layer_number, self.attention_mask)

        elif pooling == "max":
            assert (
                self.attention_mask != None
            ), "Please provide attention mask if you are using max pooling"
            pooler = Pooling("max")
            return pooler(hidden_states, layer_number, self.attention_mask)
        else:
            raise ValueError("Wrong pooling method provided in the function call")

    def get_embedding_layer_output(self, pooling="cls"):
        return self.get_embedding_from_layer(0, pooling).numpy()

    def get_embedding_last_hidden_layer(self, pooling="cls"):
        return self.get_embedding_from_layer(-1, pooling).numpy()

    def get_embedding_sum_all_layers(self, pooling="cls"):
        outputs = []
        for layer in range(13):
            output = self.get_embedding_from_layer(layer, pooling)
            outputs.append(output)
        outputs_sum = sum(outputs)
        return outputs_sum.numpy()

    def get_embedding_second_last_layer(self, pooling="cls"):
        return self.get_embedding_from_layer(-2, pooling).numpy()

    def get_embedding_sum_last_four_layers(self, pooling="cls"):
        outputs = []
        layers = [-4, -3, -2, -1]
        for layer in layers:
            output = self.get_embedding_from_layer(layer, pooling)
            outputs.append(output)
        outputs_sum = sum(outputs)
        return outputs_sum.numpy()

    def get_embedding_concat_last_four_layers(self, pooling="cls"):
        outputs = []
        layers = [-4, -3, -2, -1]
        for layer in layers:
            output = self.get_embedding_from_layer(layer, pooling)
            outputs.append(output)
        return torch.cat(outputs, dim=-1).numpy()

    def retrieve_embeddings(self):
        poolings = ["mean", "cls", "max"]
        embeddings_dict = {
            function.__name__ + "_" + pooling: []
            for function in self.functions
            for pooling in poolings
        }
        for batch in tqdm(self.dataloader, desc="Retrieving embeddings"):
            self.tokenize_and_produce_model_output(batch)
            for function in self.functions:
                for pooling in poolings:
                    embedding = function(pooling=pooling)
                    embeddings_dict[function.__name__ + "_" + pooling].extend(embedding)
                    del embedding
                    gc.collect()
        return embeddings_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--corpus_path", help="Path to the tsv file with corpus of texts", type=str, required=True
    )
    parser.add_argument("--embedding_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.embedding_model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    data = pd.read_csv(args.corpus_path, sep="\t", header=None)
    texts = data.iloc[:, 0].tolist()[:20]
    dataset = StringDataset(texts)
    dataloader = DataLoader(dataset, batch_size=64)

    embedder = EmbeddingsRetriever(model, tokenizer, dataloader)
    results = embedder.retrieve_embeddings()
    assert len(results) == len(texts), "Length of output is not equal to the length of input"
    np.save(f"../embeddings_data/{args.dataset_name}.npy", results)
