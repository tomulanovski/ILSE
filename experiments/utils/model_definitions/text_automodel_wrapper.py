from typing import Any, List

import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from llm2vec import LLM2Vec

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from ..misc.optimal_batch_size import find_optimal_batch_size
from ..dataloaders.text_dataloader import collate as text_collate

model_types = ["cerebras",
                "Pythia",
                "mamba",
                "mamba2",
                "Medical-Llama3",
                "Llama3",
                "TinyLlama",
                "bert",
                "roberta",
                "LLM2Vec-mntp-unsup-simcse",
                "LLM2Vec-mntp-supervised",
                "LLM2Vec-mntp",
                "llama-instruct"]

cerebras_sizes = ['111M', '256M', '590M', '1.3B', '2.7B', '6.7B', '13B'] # '13b' also exists but doesnt fit in 24G for bfloat16
Pythia_sizes = ['14m', '70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b'] # '12b' also exists but doesnt fit in 24G for bfloat16
mamba_sizes = ['130m', '370m', '790m', '1.4b', '2.8b']
mamba2_sizes = ['130m', '370m', '780m', '1.3b', '2.7b']
bert_sizes = ['base', 'large']
medical_llama3_sizes = ['8B'] # its only 8B model
llama3_sizes = ['8B']
tinyllama_sizes = ['1.1B']
LLM2Vec_sizes = ['8B']
llama_instruct_sizes = ['8B']

model_name_to_sizes = {
    'Pythia': Pythia_sizes,
    'cerebras': cerebras_sizes,
    'mamba': mamba_sizes,
    'mamba2': mamba2_sizes,
    'Medical-Llama3': medical_llama3_sizes,
    'Llama3': llama3_sizes,
    'TinyLlama': tinyllama_sizes,
    'bert': bert_sizes,
    'roberta': bert_sizes,
    'LLM2Vec-mntp-unsup-simcse': LLM2Vec_sizes,
    'llama-instruct': llama_instruct_sizes,
    'LLM2Vec-mntp-supervised': LLM2Vec_sizes,
    'LLM2Vec-mntp': LLM2Vec_sizes,
}


def get_model_path(name, size):
    assert name in model_types
    if name == "cerebras":
        assert size in cerebras_sizes
        return f"cerebras/Cerebras-GPT-{size}"
    elif name == "Pythia":
        assert size in Pythia_sizes
        return f"EleutherAI/pythia-{size}"
    elif name == "Medical-Llama3":
        assert size in medical_llama3_sizes
        return f"ruslanmv/Medical-Llama3-8B"
    elif name == "Llama3":
        assert size in llama3_sizes
        return f"meta-llama/Meta-Llama-3-8B"
    elif name == "TinyLlama":
        assert size in tinyllama_sizes
        return f"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif name == "mamba":
        assert size in mamba_sizes
        return f"state-spaces/mamba-{size}-hf"
    elif name == "mamba2":
        assert size in mamba2_sizes
        return f"state-spaces/mamba2-{size}-hf" 
    elif name == "bert":
        assert size in bert_sizes
        return f"bert-{size}-uncased"
    elif name == 'roberta':
        assert size in bert_sizes
        return f"FacebookAI/roberta-{size}"
    elif name == 'LLM2Vec-mntp-unsup-simcse':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp-supervised':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == "llama-instruct":
        assert size in llama_instruct_sizes
        return f"meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        raise ValueError(f"Model type {name} not found")




class TextModelSpecifications(BaseModelSpecifications):
    def __init__(self, model_family, model_size, revision, ignore_checks=False):
        super().__init__(model_family, model_size, revision, ignore_checks)
        self.model_path_func = get_model_path

    def additional_checks(self):
        if self.revision != "main":
            # currently only supporting 14m and 410m Pythia models for non-main checkpoints
            assert self.model_family == "Pythia"
            assert self.model_size in ["14m", "410m"]
        
        assert self.model_family in model_name_to_sizes.keys(), \
            f"Model family {self.model_family} not found, available families: {model_name_to_sizes.keys()}"
        assert self.model_size in model_name_to_sizes[self.model_family], \
            f"Model size {self.model_size} not found for model family {self.model_family}, available sizes: {model_name_to_sizes[self.model_family]}"

class TextLayerwiseAutoModelWrapper(BaseLayerwiseAutoModelWrapper):
    def __init__(self,
                 model_specs: TextModelSpecifications,
                 device_map="auto",
                 evaluation_layer_idx: int = -1,
                 use_memory_efficient_hooks: bool = True,
                 output_hidden_states: bool = False):
        self.use_memory_efficient_hooks = use_memory_efficient_hooks
        self.output_hidden_states = output_hidden_states
        super().__init__(model_specs, device_map, evaluation_layer_idx)

        # Storage for hook-based encoding
        self._hooked_layer_outputs = []
        self._current_pooling_method = None
        self._current_attention_mask = None

    """
    FUNCTIONS FOR INITIALIZATION
    """
    def setup_input_processor(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token is not None

        # number of tokens the model can handle
        self.max_tokens = self.tokenizer.model_max_length

    def setup_model(self):
        # For layer 0, we need output_hidden_states for legacy method
        # Only need output_hidden_states if NOT using hooks OR if layer 0
        need_hidden_states = not self.use_memory_efficient_hooks or self.evaluation_layer_idx == 0
        
        self.config = AutoConfig.from_pretrained(self.model_path,
                                            revision=self.model_specs.revision,
                                            output_hidden_states=need_hidden_states)
        self.num_layers = self.config.num_hidden_layers + 1
        self.update_evaluation_layer()
        
        # Special handling for layer 0: need at least 1 hidden layer for forward pass
        # but we'll use output_hidden_states[0] instead of hooks
        if self.evaluation_layer_idx == 0:
            self.config.num_hidden_layers = 1  # Load 1 layer so forward pass works
        else:
            self.config.num_hidden_layers = self.evaluation_layer_idx # prevents loading all layers

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            'device_map': self.device_map
        }

        if 'llm2vec' in self.model_path.lower():
            MODEL_CLASS = LLM2Vec
            if 'unsup' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
            elif 'supervised' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            elif self.model_specs.model_family.lower() == 'llm2vec-mntp':
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            else:
                raise ValueError(f"Model family {self.model_specs.model_family} not found")
        else:
            MODEL_CLASS = AutoModelForCausalLM

        self.model = MODEL_CLASS.from_pretrained(self.model_path, **FROM_PRETRAINED_KWARGS).eval()        

    """
    FUNCTIONS FOR INFERENCE
    """
    @torch.no_grad()
    def encode(
        self,
        input_data: List[str],
        return_raw_hidden_states: bool = False,
        **kwargs: dict
    ) -> np.ndarray:
        max_sample_length = kwargs.pop("max_sample_length", 2048)
        if self.model_specs.model_family in ["bert", "roberta"]:
            max_sample_length = 512
            
        verbose = kwargs.pop("verbose", True)

        tokenized_sentences =  self.tokenizer(input_data,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=max_sample_length)

        # Use provided batch_size or find optimal batch size
        provided_batch_size = kwargs.pop("batch_size", None)
        if provided_batch_size is not None:
            optimal_batch_size = provided_batch_size
        else:
            # find optimal batch size
            # optimal_batch_size = find_optimal_batch_size(model=self._get_model_with_forward_pass(),
            #                                              number_of_samples=len(input_data),
            #                                              device=self.device,
            #                                              max_sentence_length = tokenized_sentences.input_ids.shape[1],
            #                                              verbose=verbose)
            optimal_batch_size = 64
            optimal_batch_size = min(512, optimal_batch_size)
        self.batch_size_hint = optimal_batch_size

        # create dataloader
        dataset = [{"input_ids": ids, "attention_mask": mask} 
            for ids, mask in zip(tokenized_sentences["input_ids"], 
                                tokenized_sentences["attention_mask"])]
        dataloader = DataLoader(dataset, 
                                batch_size=optimal_batch_size, 
                                shuffle=False, 
                                num_workers=8, 
                                collate_fn=text_collate)

        if return_raw_hidden_states:
            embeddings, raw_hidden_states, layerwise_encodings = self._encode_helper(dataloader, 
                                                            verbose=verbose, 
                                                            return_raw_hidden_states=return_raw_hidden_states,
                                                            **kwargs)
            return np.array(embeddings), raw_hidden_states, layerwise_encodings
        
        else:
            embeddings = self._encode_helper(dataloader, 
                                            verbose=verbose, 
                                            return_raw_hidden_states=return_raw_hidden_states,
                                            **kwargs) # shape: (num_samples, embedding_dim)
            return np.array(embeddings)
    
    
    def _get_model_with_forward_pass(self):
        if 'llm2vec' in self.model_path.lower():
            return self.model.model
        else:
            return self.model
    
    @torch.no_grad()
    def _encode_helper(self, dataloader, verbose=False, return_raw_hidden_states=False, **kwargs) -> np.ndarray:
        pooling_method = kwargs.pop("pooling_method", "mean")

        # Special case: Use legacy method for layer 0 (embedding) with Llama models
        # The hook-based method doesn't work reliably when num_hidden_layers=0
        is_llama_layer_0 = (self.evaluation_layer_idx == 0 and 
                           hasattr(self.model, 'model') and 
                           hasattr(self.model.model, 'layers'))
        
        if self.use_memory_efficient_hooks and not is_llama_layer_0:
            return self._encode_helper_with_hooks(dataloader, verbose, return_raw_hidden_states, pooling_method, **kwargs)
        else:
            return self._encode_helper_legacy(dataloader, verbose, return_raw_hidden_states, pooling_method, **kwargs)

    @torch.no_grad()
    def _encode_helper_with_hooks(self, dataloader, verbose, return_raw_hidden_states, pooling_method, **kwargs):
        """Memory-efficient encoding using forward hooks to pool layer outputs immediately."""
        # Setup for hooks
        self._current_pooling_method = pooling_method
        hooks = self._register_pooling_hooks()

        encoded_batches = []
        layerwise_encoded_batches = []

        try:
            for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable=not verbose):
                batch = self.prepare_inputs(batch)

                # Store attention mask for hooks to access
                self._current_attention_mask = batch["attention_mask"]
                self._hooked_layer_outputs = []

                # Forward pass - hooks will collect and pool layer outputs
                _ = self.forward(**batch)

                # Get the encoding from the evaluation layer
                final_encoding = self._hooked_layer_outputs[self.evaluation_layer_idx]
                encoded_batches.append(final_encoding.cpu())

                if return_raw_hidden_states:
                    # Stack all pooled layer outputs for this batch
                    batch_layerwise = torch.stack(self._hooked_layer_outputs)
                    layerwise_encoded_batches.append(batch_layerwise.cpu())

                # Clear GPU cache after each batch to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            # Always remove hooks
            for hook in hooks:
                hook.remove()

            # Cleanup
            self._hooked_layer_outputs = []
            self._current_attention_mask = None
            self._current_pooling_method = None

        # Concatenate all batches
        encodings = torch.cat(encoded_batches).squeeze()
        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)
        encodings = encodings.numpy()

        if return_raw_hidden_states:
            layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy()
            # Final GPU cache clear after all processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Note: no longer returning raw token-level states, only pooled layerwise encodings
            return encodings, None, layerwise_encodings
        else:
            # Final GPU cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return encodings

    @torch.no_grad()
    def _encode_helper_legacy(self, dataloader, verbose, return_raw_hidden_states, pooling_method, **kwargs):
        """Legacy encoding using output_hidden_states (kept for backward compatibility)."""
        encoded_batches = []
        layerwise_encoded_batches = []

        if return_raw_hidden_states:
            # can be memory intensive, so only do if needed
            raw_sample_hidden_states = []

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable=not verbose):
            batch = self.prepare_inputs(batch)

            outputs = self.forward(**batch)

            hidden_states = outputs.hidden_states[self.evaluation_layer_idx]

            hidden_states = self._get_pooled_hidden_states(hidden_states, batch["attention_mask"], method=pooling_method)
            encoded_batches.append(hidden_states.float().cpu())

            if return_raw_hidden_states:
                # get layerwise encodings for the batch
                current_batch_layerwise_encodings = []
                for layer_idx in range(len(outputs.hidden_states)):
                    layer_states = outputs.hidden_states[layer_idx]


                    layer_states = self._get_pooled_hidden_states(layer_states, batch["attention_mask"], method=pooling_method)
                    current_batch_layerwise_encodings.append(layer_states.float().cpu())
                layerwise_encoded_batches.append(torch.stack(current_batch_layerwise_encodings))

                # get raw hidden states for each sample
                for sample_idx in range(len(outputs.hidden_states[0])):
                    pad_idx = batch['attention_mask'][sample_idx] == 0

                    sample_hidden_states = [
                        layer_states[sample_idx][~pad_idx]
                        for layer_states in outputs.hidden_states
                    ]
                    sample_hidden_states = torch.stack(sample_hidden_states)
                    raw_sample_hidden_states.append(sample_hidden_states.squeeze().float().cpu().numpy())

        encodings = torch.cat(encoded_batches).squeeze() # shape: (num_samples, embedding_dim)
        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)
        encodings = encodings.numpy()

        if return_raw_hidden_states:
            layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy() # shape: (num_layers, num_samples, embedding_dim)
            return encodings, raw_sample_hidden_states, layerwise_encodings
        else:
            return encodings
    
    @torch.no_grad()
    # TODO: check if first and last are real and not padding or special tokens
    def _get_pooled_hidden_states(self, hidden_states, attention_mask=None, method="mean"):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states[0])

        if method == "mean":
            seq_lengths = attention_mask.sum(dim=-1)
            return torch.stack(
                [
                    hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif method == "mean_including_padding":
            layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states])
            return layer_means
        
        elif method == "last_hidden_state":
            return hidden_states[:, -1]
        elif method == "first_hidden_state":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {method}")
        
    def prepare_inputs(self, batch):
        # move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # squeeze if needed
        if len(batch['input_ids'].shape) == 3:
            batch = {k: v.squeeze() for k, v in batch.items()}

        # unsqueeze if needed, such as for augmentation dataloaders
        if len(batch['input_ids'].shape) == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        return batch

    """
    MEMORY-EFFICIENT FORWARD HOOK FUNCTIONS
    """
    def _get_layer_modules(self):
        """Get the list of layer modules to hook for collecting hidden states.
        Returns a tuple: (embedding_module, transformer_layer_modules, final_layer_norm_module)
        """
        model = self._get_model_with_forward_pass()

        # Detect model architecture and get appropriate layers
        if hasattr(model, 'gpt_neox'):  # Pythia (GPT-NeoX)
            embedding = model.gpt_neox.embed_in
            layers = list(model.gpt_neox.layers[:self.evaluation_layer_idx])
            final_layer_norm = model.gpt_neox.final_layer_norm

        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # GPT-2 style (Cerebras)
            embedding = model.transformer.wte  # word token embeddings
            layers = list(model.transformer.h[:self.evaluation_layer_idx])
            final_layer_norm = model.transformer.ln_f  # GPT-2 uses ln_f for final layer norm

        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Llama
            embedding = model.model.embed_tokens
            layers = list(model.model.layers[:self.evaluation_layer_idx])
            final_layer_norm = model.model.norm  # Llama uses 'norm' for final layer norm

        elif hasattr(model, 'bert'):  # BERT
            embedding = model.bert.embeddings
            layers = list(model.bert.encoder.layer[:self.evaluation_layer_idx])
            final_layer_norm = None  # BERT doesn't have a final layer norm after encoder

        elif hasattr(model, 'roberta'):  # RoBERTa
            embedding = model.roberta.embeddings
            layers = list(model.roberta.encoder.layer[:self.evaluation_layer_idx])
            final_layer_norm = None  # RoBERTa doesn't have a final layer norm after encoder

        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):  # Mamba
            embedding = model.backbone.embeddings if hasattr(model.backbone, 'embeddings') else None
            layers = list(model.backbone.layers[:self.evaluation_layer_idx])
            final_layer_norm = model.backbone.norm_f if hasattr(model.backbone, 'norm_f') else None

        else:
            raise NotImplementedError(
                f"Layer extraction not implemented for this model architecture. "
                f"Model type: {type(model)}, available attributes: {dir(model)}"
            )

        return embedding, layers, final_layer_norm

    def _make_pooling_hook(self, layer_idx, final_layer_norm=None):
        """Create a forward hook that pools hidden states and stores them.

        Args:
            layer_idx: Index of this layer in our output list
            final_layer_norm: If provided, apply this normalization before pooling (for last layer)
        """
        def hook_fn(module, input, output):
            # Extract hidden states from output
            # Different architectures return different formats
            if isinstance(output, tuple):
                hidden_states = output[0]  # Usually (hidden_states, ...) for transformers
            else:
                hidden_states = output

            # Apply final layer norm if this is the last layer
            # This ensures hook-based encoding matches output_hidden_states behavior
            if final_layer_norm is not None:
                hidden_states = final_layer_norm(hidden_states)

            # Pool the hidden states immediately
            pooled = self._get_pooled_hidden_states(
                hidden_states,
                self._current_attention_mask,
                method=self._current_pooling_method
            )

            # Store pooled output (much smaller than full token representation)
            self._hooked_layer_outputs.append(pooled.float())

            return output  # Return original output for model to continue

        return hook_fn

    def _register_pooling_hooks(self):
        """Register forward hooks on all layers to pool outputs immediately."""
        embedding, layers, final_layer_norm = self._get_layer_modules()
        hooks = []

        # Hook embedding layer (this will be layer 0 in our outputs)
        if embedding is not None:
            hook = embedding.register_forward_hook(self._make_pooling_hook(0))
            hooks.append(hook)

        # Hook transformer layers (these will be layers 1, 2, ..., L)
        for layer_idx, layer in enumerate(layers):
            # For the last transformer layer, apply final_layer_norm before pooling
            # This ensures hook-based encoding matches output_hidden_states behavior
            is_last_layer = (layer_idx == len(layers) - 1)
            norm_to_apply = final_layer_norm if is_last_layer else None

            hook = layer.register_forward_hook(self._make_pooling_hook(layer_idx + 1, norm_to_apply))
            hooks.append(hook)

        return hooks