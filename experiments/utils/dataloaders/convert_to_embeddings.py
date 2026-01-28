import torch
import tqdm
import numpy as np
import os

def convert_image_dataset_to_embeddings(dataloader, model, save_path):
    embeddings = []
    labels = []

    with torch.no_grad():
        model.model.eval()

        for batch in tqdm.tqdm(dataloader, desc=f"Converting set to embeddings"):
            x, y = model.prepare_inputs(batch, return_labels=True)
            outputs = model(**x)
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif isinstance(outputs, dict) and 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs
            
            layerwise_feats = []
            for layer_idx, layer in enumerate(hidden_states):
                pooled_feats = model._get_pooled_hidden_states(layer, None, method="mean") # [batch_size, hidden_size]
                layerwise_feats.append(pooled_feats.cpu().half().detach().squeeze().numpy())
            
            layerwise_feats = np.array(layerwise_feats).swapaxes(0, 1)
            embeddings.extend(layerwise_feats)
            labels.extend(y.cpu().detach().squeeze().numpy())
    
    # Save embeddings and labels separately to avoid loading full arrays into memory
    embeddings_path = save_path + '.embeddings.npy'
    labels_path = save_path + '.labels.npy'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save numpy arrays directly to disk
    np.save(embeddings_path, np.array(embeddings))
    np.save(labels_path, np.array(labels))

    # Create a custom dataset that loads data lazily
    class LazyLoadDataset(torch.utils.data.Dataset):
        def __init__(self, embeddings_path, labels_path):
            self.embeddings = np.load(embeddings_path, mmap_mode='r')  # Memory-mapped read
            self.labels = np.load(labels_path, mmap_mode='r')
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return torch.from_numpy(self.embeddings[idx]).half(), torch.tensor(self.labels[idx], dtype=torch.int64)

    dataset = LazyLoadDataset(embeddings_path, labels_path)
    print(f"Dataset size: {len(dataset)} samples")
    return dataset


