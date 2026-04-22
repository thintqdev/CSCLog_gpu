"""
GPU-Accelerated Embedding Generator for CSCLog
Uses BERT to generate sentence embeddings with V100 optimization
"""

import torch
import numpy as np
import json
from typing import Dict, List
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import gc


class GPUError(Exception):
    """GPU-related errors"""
    pass


class EmbeddingGenerator:
    """Generate sentence embeddings using GPU-accelerated BERT"""
    
    def __init__(self, model_name: str = "model/bert", batch_size: int = 128,
                 max_batch_size: int = 256, device: str = "cuda",
                 max_length: int = 512, use_fp16: bool = True,
                 use_tf32: bool = True, pin_memory: bool = True):
        """
        Initialize BERT model on specified device
        
        Args:
            model_name: Path to BERT model or HuggingFace model name
            batch_size: Initial batch size
            max_batch_size: Maximum batch size
            device: 'cuda' or 'cpu'
            max_length: Maximum sequence length
            use_fp16: Use mixed precision (FP16) for V100 Tensor Cores
            use_tf32: Use TF32 for additional V100 speedup
            pin_memory: Pin memory for faster CPU-GPU transfers
        """
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.pin_memory = pin_memory
        
        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
            self.use_fp16 = False  # FP16 only works on GPU
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Enable TF32 for V100 if requested
        if use_tf32 and device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for V100 acceleration")
        
        # Load tokenizer and model
        print(f"Loading BERT model from {model_name}...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Convert to FP16 if requested
            if self.use_fp16 and device == "cuda":
                self.model = self.model.half()
                print("Model converted to FP16 for Tensor Core acceleration")
            
            print(f"Model loaded successfully on {self.device}")
            
            # Warmup to optimize first batch
            self._warmup()
            
        except Exception as e:
            raise GPUError(f"Failed to load BERT model: {e}")
    
    def _warmup(self):
        """Warmup model to optimize first batch"""
        print("Warming up model...")
        dummy_text = ["This is a warmup text"] * min(self.batch_size, 32)
        try:
            with torch.no_grad():
                self._batch_encode(dummy_text)
            print("Warmup complete")
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")
    
    def generate_embeddings(self, templates: Dict[str, str], 
                          progress_bar: bool = True) -> Dict[str, List[float]]:
        """
        Generate embeddings for all templates with GPU batching
        
        Args:
            templates: Dict mapping EventId -> template_text
            progress_bar: Show progress bar
            
        Returns:
            embeddings: Dict mapping EventId -> 768-dim vector
        """
        print(f"Generating embeddings for {len(templates)} templates...")
        print(f"Initial batch size: {self.batch_size}")
        
        embeddings = {}
        event_ids = list(templates.keys())
        texts = [templates[eid] for eid in event_ids]
        
        current_batch_size = self.batch_size
        i = 0
        
        iterator = tqdm(total=len(texts), desc="Generating embeddings") if progress_bar else None
        
        while i < len(texts):
            try:
                # Get batch
                batch_ids = event_ids[i:i+current_batch_size]
                batch_texts = texts[i:i+current_batch_size]
                
                # Generate embeddings
                batch_embeddings = self._batch_encode(batch_texts)
                
                # Store results
                for eid, emb in zip(batch_ids, batch_embeddings):
                    embeddings[eid] = emb.tolist()
                
                i += current_batch_size
                
                if iterator:
                    iterator.update(len(batch_texts))
                
                # Try to increase batch size if GPU memory < 70%
                if self.device.type == "cuda":
                    memory_usage = self.get_gpu_memory_usage()
                    if memory_usage < 0.7 and current_batch_size < self.max_batch_size:
                        new_batch_size = min(int(current_batch_size * 1.5), self.max_batch_size)
                        if new_batch_size > current_batch_size:
                            current_batch_size = new_batch_size
                            if iterator:
                                iterator.set_postfix({"batch_size": current_batch_size, 
                                                    "gpu_mem": f"{memory_usage:.1%}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM error - reduce batch size
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    current_batch_size = max(current_batch_size // 2, 1)
                    print(f"\nGPU OOM! Reducing batch size to {current_batch_size}")
                    
                    if current_batch_size < 1:
                        raise GPUError("Cannot process even single sample. GPU memory insufficient.")
                    
                    continue  # Retry with smaller batch
                else:
                    raise GPUError(f"GPU operation failed: {e}")
        
        if iterator:
            iterator.close()
        
        print(f"Embedding generation complete. Final batch size: {current_batch_size}")
        
        return embeddings
    
    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode batch of texts on GPU
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings [batch_size, 768]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        if self.pin_memory and self.device.type == "cuda":
            input_ids = encoded['input_ids'].pin_memory().to(self.device, non_blocking=True)
            attention_mask = encoded['attention_mask'].pin_memory().to(self.device, non_blocking=True)
        else:
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate embeddings (no gradients needed)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Move back to CPU and convert to float32 if needed
        if self.use_fp16:
            embeddings = embeddings.float()
        
        return embeddings.cpu()
    
    def save_embeddings(self, embeddings: Dict[str, List[float]], output_path: str):
        """
        Save embeddings to JSON file
        
        Args:
            embeddings: Dict mapping EventId -> embedding vector
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings saved to {output_path}")
    
    def get_gpu_memory_usage(self) -> float:
        """
        Return current GPU memory usage as percentage
        
        Returns:
            Memory usage (0.0 to 1.0)
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            return allocated / total
        return 0.0
    
    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """
        Return GPU memory stats
        
        Returns:
            Dict with 'used_mb', 'total_mb', 'percentage'
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)  # MB
            percentage = allocated / total
            
            return {
                'used_mb': allocated,
                'total_mb': total,
                'percentage': percentage
            }
        return {'used_mb': 0, 'total_mb': 0, 'percentage': 0}
    
    def __del__(self):
        """Cleanup GPU memory on deletion"""
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
