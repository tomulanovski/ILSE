import os
import math
import warnings

import numpy as np
import torch
import tqdm
import umap
from datasets import Dataset, load_dataset, load_from_disk
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

datasets = ['wikitext', 'ai-medical-dataset']

def find_data_key_in_examples(examples):
    if "text" in examples:
        return "text"
    elif "sentences" in examples:
        return "sentences"
    elif "query" in examples:
        return "query"
    elif "sentence1" in examples and "sentence2" in examples:
        return "sentence1"
    else:
        raise ValueError("No text or sentences column found in examples, valid columns: ", examples.keys())
        
def get_dataloader(
        tokenizer, 
        dataset_name, 
        split='train', 
        context_length_ratio=1, 
        min_length=5,
        max_length=None, 
        num_samples=10000, 
        filter_text_columns=True, 
        augment=False,
        return_dataset=False,
        max_sample_length=2048,
        num_workers=8,
        batch_size=1
    ):

    def general_tokenize_function(examples):
        data_key = find_data_key_in_examples(examples)
        sentences = examples[data_key]
        if isinstance(sentences[0], list):
            sentences = [item for sublist in sentences for item in sublist]

        if not augment:
            texts = sentences
        else:
            texts = text_augmentation(sentences) 

        return tokenizer(texts, truncation=True, max_length=max_sample_length)
    
    def medical_tokenize_function(examples):
        medical_prompt = """You are an AI Medical Assistant Chatbot, trained to answer medical questions. Below is an instruction that describes a task, paired with an response context. Write a response that appropriately completes the request.

            ### Instruction:
            {}


            ### Response:
            {}"""
        
        instructions = examples["question"]
        outputs      = examples["context"]
        texts = []
        for instruction, output in zip(instructions,  outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = medical_prompt.format(instruction,  output)
            texts.append(text)

        return tokenizer(texts, truncation=True, max_length=max_sample_length)
    
    def adjust_context_length(examples):
        if context_length_ratio == 1:
            return examples
        else:
            input_length = len(examples['input_ids'])
            context_length = max(2, int(input_length * context_length_ratio))
            examples['attention_mask'] = examples['attention_mask'][:context_length]
            examples['input_ids'] = examples['input_ids'][:context_length]

            return examples

    def is_not_wikipedia_heading(example):
        return not (example["text"].strip().startswith("=") and example["text"].strip().endswith("="))

    assert dataset_name in datasets or 'mteb' in dataset_name
    assert context_length_ratio <= 1

    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", 'wikitext-103-v1')[split]
    
        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
        dataset = dataset.filter(is_not_wikipedia_heading) # filter out headings
        
        # filter out samples by lower bound and upper bound on length
        dataset = dataset.filter(lambda x: len(x['text']) >= 2*min_length) # filter out the frequent blank/small examples in the dataset
        if max_length is not None:
            dataset = dataset.filter(lambda x: len(x['text']) <= 2*max_length)

        # tokenize the dataset
        try:
            tokenized_dataset = dataset.map(general_tokenize_function, batched=True).shuffle(seed=42)
            tokenized_dataset.set_format("torch")
        except Exception as e:
            for idx, d in enumerate(dataset):
                print(idx, d)
            raise e
        
        if filter_text_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    elif dataset_name == 'ai-medical-dataset':
        dataset = load_dataset("ruslanmv/ai-medical-dataset")[split]
    
        # filter out unneeded samples
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

        # tokenize the dataset
        tokenized_dataset = dataset.map(medical_tokenize_function, batched=True).shuffle(seed=42)
        tokenized_dataset.set_format("torch")

        if filter_text_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(["question"])
            tokenized_dataset = tokenized_dataset.remove_columns(["context"])

        # filter out samples by lower bound and upper bound on length
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= min_length) # filter out the frequent blank/small examples in the dataset
        if max_length is not None:
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) <= max_length)

    elif 'mteb' in dataset_name:
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=True)[split]
        except KeyError as e:
            raise KeyError(f"SplitDoesNotExist: The dataset {dataset_name} does not have split {split}. Raising error to skip this dataset/split")
        except Exception as e:
            print(f"Failed to load dataset {dataset_name} with split {split} with error {e}")
            raise e

        data_key = find_data_key_in_examples(dataset[0])
        if isinstance(dataset[0][data_key], list):
            # data is splits, choose the first split
            sentences = [item for item in dataset[0][data_key]]
            dataset = Dataset.from_dict({"text": sentences})

        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

        tokenized_dataset = dataset.map(general_tokenize_function, batched=True).shuffle(seed=42)
        tokenized_dataset.set_format("torch")

        if filter_text_columns:
            for column in tokenized_dataset.column_names:
                if column not in ['input_ids', 'attention_mask']:
                    tokenized_dataset = tokenized_dataset.remove_columns([column])


    # if context_length_ratio < 1, reduce all sentences to that ratio of length
    tokenized_dataset = tokenized_dataset.map(adjust_context_length, batched=False)

    if return_dataset:
        return tokenized_dataset
    
    # form dataloader
    dataloader = DataLoader(tokenized_dataset, 
                            shuffle=False, 
                            num_workers=num_workers, 
                            batch_size=batch_size,
                            collate_fn=collate)
    return dataloader

def multiview_collate(batch, tokenizer, max_sample_length=2048, num_views=8):
    """
    Collates and augments each sample in the batch multiple times.
    Returns a list of augmented batches.
    """
    # Augment each sample in the batch num_views times
    augmented_batches = []
    for _ in range(num_views):
        augmented_batch = []
        for item in batch:
            # Get text from input_ids
            data_key = find_data_key_in_examples(item)
            text = item[data_key]
            # Augment single text
            augmented_text = text_augmentation([text], num_augmentations_per_sample=1)[0]
            # Tokenize back
            augmented_tokens = tokenizer(augmented_text, truncation=True, max_length=max_sample_length)
            augmented_batch.append({
                'input_ids': torch.tensor(augmented_tokens['input_ids']),
                'attention_mask': torch.tensor(augmented_tokens['attention_mask'])
            })
        
        # Collate the augmented batch
        collated = collate(augmented_batch)
        augmented_batches.append(collated)
    
    return tuple(augmented_batches)

def get_augmentation_collated_dataloader(
        tokenizer, 
        dataset_name, 
        split='train',
        num_augmentations_per_sample=8,
        context_length_ratio=1, 
        min_length=2,
        max_length=None, 
        num_samples=10000, 
        filter_text_columns=False,
        max_sample_length=2048,
        num_workers=8,
        batch_size=1,
    ):
    # Get base dataset without augmentation
    base_dataset = get_dataloader(
        tokenizer, 
        dataset_name, 
        split=split, 
        context_length_ratio=context_length_ratio, 
        min_length=min_length,
        max_length=max_length, 
        num_samples=num_samples, 
        filter_text_columns=False, 
        augment=False,
        return_dataset=True,
        max_sample_length=max_sample_length,
        num_workers=num_workers,
        batch_size=batch_size
    )

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        base_dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=lambda batch: multiview_collate(batch, tokenizer, max_sample_length=max_sample_length, num_views=num_augmentations_per_sample)
    )

    return dataloader


def embed_sentences_and_get_outputs(model, tokenizer, sentences: list[str]):
    tokenized_string= tokenizer(sentences, truncation=False, return_tensors='pt')
    tokenized_string = {k: v.to(model.device) for k, v in tokenized_string.items()}
    with torch.no_grad():
        outputs = model(**tokenized_string)
    
    outputs['input_ids'] = list(tokenized_string['input_ids'])
    return outputs


def reduce_and_visualize_hidden_states(hidden_states, reduction="tsne", labels=None):
    assert reduction in ["tsne", "umap"]
    warnings.filterwarnings(action='ignore', category=UserWarning)

    layers_per_row = 5
    column_width, row_height = 3, 3

    num_layers = len(hidden_states)
    num_rows = math.ceil(num_layers / layers_per_row)
    fig, axs = plt.subplots(num_rows, layers_per_row, figsize=(row_height*layers_per_row, column_width*num_rows))
    num_tokens = hidden_states[0].shape[1]

    print("NUM LAYERS", num_layers)

    # reduce and plot hidden states at each layer
    # go in reverse to make sure that dimensionality reduction has good initialization
    reduced_embeddings_by_layer = []
    for i in tqdm.tqdm(list(reversed(range(num_layers)))):
        row, col = divmod(i, layers_per_row)

        layer_hidden_states = hidden_states[i].squeeze().cpu().numpy()

        if reduction == "tsne":
            if len(reduced_embeddings_by_layer):
                # for some consistency between layers
                tsne_reducer = TSNE(n_components=2, perplexity=20, random_state=0, metric="cosine", init=reduced_embeddings_by_layer[-1])
            else:
                tsne_reducer = TSNE(n_components=2, perplexity=20, random_state=0, metric="cosine", init="pca")
            reduced_results = tsne_reducer.fit_transform(layer_hidden_states)
            reduced_embeddings_by_layer.append(reduced_results)
        elif reduction == "umap":
            if len(reduced_embeddings_by_layer):
                # for some consistency between layers
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=0, init=reduced_embeddings_by_layer[-1])
            else:
                umap_reducer = umap.UMAP(n_components=2, n_neighbors=5, random_state=0, init="spectral")
            reduced_results = umap_reducer.fit_transform(layer_hidden_states)
            reduced_embeddings_by_layer.append(reduced_results)

        if labels is None:          
            colors = np.array(list(range(num_tokens)))
        else:
            colors = labels
        
        # plot reduced embeddings
        axs[row][col].scatter(reduced_results[:, 0], reduced_results[:, 1], c=colors, cmap="viridis")
        axs[row][col].text(0.95, 0.95, f"Layer {i}", transform=axs[row][col].transAxes, ha="left", va="top") # put row number in corner
        axs[row][col].axis("off")   # hide axes


    # hide empty plots
    for i in range(num_layers, num_rows*layers_per_row):
        row, col = divmod(i, layers_per_row)
        axs[row][col].axis("off")

    fig.show()

    # unreverse the reduced embeddings
    reduced_embeddings_by_layer = list(reversed(reduced_embeddings_by_layer))
    return reduced_embeddings_by_layer

def text_augmentation(texts, num_augmentations_per_sample=1):
    # input is list of strings
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    import nlpaug.flow as naf

    aug = naf.Sequential([
        naw.SplitAug(),
        nac.RandomCharAug(),
        nac.KeyboardAug()
    ])

    augmented_text = [str(aug.augment(x, n=num_augmentations_per_sample)) for x in texts]

    return augmented_text


def collate(batch):
    ips = [item['input_ids'] for item in batch]
    attn = [item['attention_mask'] for item in batch]

    # pad to max length
    max_length = max([len(ip) for ip in ips])
    ips = [torch.nn.functional.pad(ip, (0, max_length - len(ip))) for ip in ips]
    attn = [torch.nn.functional.pad(ip, (0, max_length - len(ip))) for ip in attn]

    return {'input_ids': torch.stack(ips),
            'attention_mask': torch.stack(attn),
            }