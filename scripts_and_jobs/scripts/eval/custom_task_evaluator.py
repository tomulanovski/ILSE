"""
Custom task evaluation for datasets not in MTEB.
Compatible with MTEB-style evaluation workflow.
Loads data from HuggingFace using MTEB task metadata.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import mteb
from datasets import load_dataset


class CustomMTEBTask:
    """
    Custom task that mimics MTEB's interface.
    Use this to evaluate on datasets not in MTEB.
    """
    
    def __init__(self, task_name: str, train_data: Dict, test_data: Dict):
        """
        Args:
            task_name: Name of your task
            train_data: Dict with 'text' and 'labels'
            test_data: Dict with 'text' and 'labels'
        """
        self.task_name = task_name
        self.train_data = train_data
        self.test_data = test_data
        
        # MTEB-like metadata
        self.metadata = type('obj', (object,), {
            'name': task_name,
            'type': 'Classification',
            'description': f'Custom classification task: {task_name}'
        })()
    
    def evaluate(self, model, split="test", batch_size=32, **kwargs):
        """
        Evaluate model on this task (MTEB-compatible interface).
        
        Args:
            model: Model with encode() method
            split: 'test' or 'validation'
            batch_size: Batch size for encoding
            
        Returns:
            Dict with metrics in MTEB format
        """
        # Get data
        train_texts = self.train_data['text']
        train_labels = self.train_data['labels']
        test_texts = self.test_data['text']
        test_labels = self.test_data['labels']
        
        # Encode
        print(f"[{self.task_name}] Encoding {len(train_texts)} train samples...")
        train_embeddings = model.encode(
            train_texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            **kwargs
        )
        
        print(f"[{self.task_name}] Encoding {len(test_texts)} test samples...")
        test_embeddings = model.encode(
            test_texts,
            batch_size=batch_size, 
            show_progress_bar=True,
            **kwargs
        )
        
        # Train classifier
        print(f"[{self.task_name}] Training classifier...")
        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(train_embeddings, train_labels)
        
        # Predict
        predictions = clf.predict(test_embeddings)
        
        # Metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1_micro = f1_score(test_labels, predictions, average='micro')
        f1_macro = f1_score(test_labels, predictions, average='macro')
        
        # Return in MTEB format
        results = {
            self.task_name: {
                'test': {
                    'accuracy': accuracy,
                    'f1': f1_macro,
                    'f1_micro': f1_micro,
                }
            }
        }
        
        print(f"\n[{self.task_name}] Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        
        return results


def load_task_data_from_hf(task_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load train and test data for a task from HuggingFace using MTEB metadata.
    
    Args:
        task_name: MTEB task name
        
    Returns:
        Dict with 'train' and 'test' keys, each containing 'text' and 'labels'
        Example: {'train': {'text': [...], 'labels': [...]}, 
                  'test': {'text': [...], 'labels': [...]}}
    """    
    task_name = task_name.strip()
    
    try:
        task = mteb.get_task(task_name)
        
        # Get HuggingFace dataset info from metadata
        dataset_info = getattr(task.metadata, 'dataset', None)
        
        if not dataset_info:
            raise ValueError(f"No HuggingFace dataset info found for task '{task_name}'")
        
        # Extract dataset path and config
        if isinstance(dataset_info, dict):
            hf_path = dataset_info.get('path')
            hf_config = dataset_info.get('name')
            hf_revision = dataset_info.get('revision')
        elif isinstance(dataset_info, str):
            hf_path = dataset_info
            hf_config = None
            hf_revision = None
        else:
            raise ValueError(f"Invalid dataset info format for task '{task_name}'")
        
        print(f"Loading {task_name} from HuggingFace: {hf_path}")
        if hf_config:
            print(f"  Config: {hf_config}")
        
        # Load dataset from HuggingFace
        load_kwargs = {}
        if hf_config:
            load_kwargs['name'] = hf_config
        if hf_revision:
            load_kwargs['revision'] = hf_revision
            
        dataset = load_dataset(hf_path, **load_kwargs)
        
        # Check for English language key
        if 'en' in dataset:
            print(f"Found 'en' language key")
            dataset = dataset['en']
        elif 'eng' in dataset:
            print(f"Found 'eng' language key")
            dataset = dataset['eng']
        
        # Load both train and test splits
        result = {}
        
        for split_name in ['train', 'test']:                                   
            
            if split_name not in dataset:
                available_splits = list(dataset.keys())
                raise ValueError(f"Split '{split_name}' not available for task '{task_name}'. Available splits: {available_splits}")
            
            split_data = dataset[split_name]
            print(f"Loaded {len(split_data)} examples from {split_name} split")
            
            # Extract text and labels
            columns = split_data.column_names
            
            # choose proper keyword for the samples data
            if "text" in columns:
                text_keyword = "text"
            elif "verse_text" in columns:
                text_keyword = "verse_text"
            else:
                raise ValueError(f"Dataset has unknown 'text' like column. Found: {columns}")

            if "label" in columns:
                label_keyword = "label"
            else:
                raise ValueError(f"Dataset has unknown 'label' like column. Found: {columns}")
                                    
            result[split_name] = {
                'text': split_data[text_keyword],
                'labels': split_data[label_keyword]
            }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error loading task '{task_name}': {e}")


def load_custom_task(task_name: str) -> Optional[CustomMTEBTask]:
    """
    Load a custom task by name using HuggingFace datasets via MTEB metadata.
    
    Args:
        task_name: Name of the custom task (must be in MTEB)
        
    Returns:
        CustomMTEBTask instance or None if not found
    """
    try:
        print(f"Loading custom task data for: {task_name}")
        
        # Load both train and test data at once
        data = load_task_data_from_hf(task_name)
        
        train_data = data['train']
        test_data = data['test']
        
        return CustomMTEBTask(task_name, train_data, test_data)
        
    except Exception as e:
        print(f"Warning: Could not load custom task '{task_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_custom_tasks(model, custom_task_names, batch_size=32):
    """
    Evaluate model on multiple custom tasks.
    
    Args:
        model: Model with encode() method
        custom_task_names: List of custom task names
        batch_size: Batch size for encoding
        
    Returns:
        Dict with results for all custom tasks
    """
    all_results = {}
    
    print("\n" + "="*60)
    print("EVALUATING CUSTOM TASKS")
    print("="*60)
    
    for task_name in custom_task_names:
        try:
            print(f"\nLoading custom task: {task_name}")
            custom_task = load_custom_task(task_name)
            
            if custom_task is None:
                print(f"✗ Failed to load custom task: {task_name}")
                continue
            
            print(f"Evaluating {task_name}...")
            results = custom_task.evaluate(model, split="test", batch_size=batch_size)
            
            if results:
                all_results.update(results)
                print(f"✓ {task_name} evaluation completed")
            
        except Exception as e:
            print(f"✗ Error evaluating custom task {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results