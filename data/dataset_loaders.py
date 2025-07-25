"""
Dataset loaders for DYNAMO training.
Handles loading and preprocessing of SST-2, SQuAD, XSum, code generation, and translation datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import RobertaTokenizer
from typing import Dict, List, Optional, Tuple, Any, Union
import random
import json
import os
import hashlib

from utils.logger import get_logger
from utils.helpers import set_seed

logger = get_logger(__name__)


class DynamoDataset(Dataset):
    """
    Base dataset class for DYNAMO tasks.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        task_name: str = "unknown"
    ):
        """
        Initialize DYNAMO dataset.
        
        Args:
            data: List of data examples
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
            task_name: Name of the task
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.data[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare output
        output = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': self.task_name,
            'task_id': example.get('task_id', 0)
        }
        
        # Add task-specific targets
        if 'target' in example:
            output['target'] = example['target']
        
        return output


class SentimentDataset(DynamoDataset):
    """Dataset for sentiment analysis (SST-2)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input
        tokenized = self.tokenizer(
            example['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Validate and clamp label to valid range [0, 1]
        label = example['label']
        if not isinstance(label, int) or label < 0 or label > 1:
            # Default to neutral/positive if invalid
            label = 1
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'sentiment',
            'task_id': 0,
            'target': torch.tensor(label, dtype=torch.long),
            'input_text': example['sentence']
        }


class QADataset(DynamoDataset):
    """Dataset for question answering (SQuAD)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Combine question and context
        input_text = f"{example['question']} [SEP] {example['context']}"
        
        # Tokenize input
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get answer positions and clamp to valid range
        start_pos = example.get('start_position', 0)
        end_pos = example.get('end_position', 0)
        
        # Clamp positions to valid range [0, max_length-1]
        max_pos = self.max_length - 1
        start_pos = max(0, min(start_pos, max_pos))
        end_pos = max(start_pos, min(end_pos, max_pos))
        
        # Ensure end_pos >= start_pos
        if end_pos < start_pos:
            end_pos = start_pos
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'qa',
            'task_id': 1,
            'target': torch.tensor([start_pos, end_pos], dtype=torch.long),
            'input_text': input_text
        }


class SummarizationDataset(DynamoDataset):
    """Dataset for summarization (XSum)."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input document
        tokenized = self.tokenizer(
            example['document'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create deterministic target representation based on summary text
        # Use hash of summary text to create consistent representation
        summary_hash = hashlib.md5(example['summary'].encode()).hexdigest()
        # Convert hash to seed for reproducible random generation
        seed = int(summary_hash[:8], 16)
        torch.manual_seed(seed)
        target_representation = torch.randn(768)  # Deterministic based on summary
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'summarization',
            'task_id': 2,
            'target': target_representation,  # [768] deterministic representation
            'target_text': example['summary'],  # Keep original for reference
            'input_text': example['document']
        }


class CodeGenerationDataset(DynamoDataset):
    """Dataset for code generation."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize input problem
        tokenized = self.tokenizer(
            example['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create deterministic target representation based on code text
        # Use hash of code text to create consistent representation
        code_hash = hashlib.md5(example['code'].encode()).hexdigest()
        # Convert hash to seed for reproducible random generation
        seed = int(code_hash[:8], 16)
        torch.manual_seed(seed)
        target_representation = torch.randn(768)  # Deterministic based on code
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'code_generation',
            'task_id': 3,
            'target': target_representation,  # [768] deterministic representation
            'target_text': example['code'],  # Keep original for reference
            'input_text': example['text']
        }


class TranslationDataset(DynamoDataset):
    """Dataset for translation."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        
        # Tokenize source text (English)
        tokenized = self.tokenizer(
            example['en'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create deterministic target representation based on German text
        # Use hash of German text to create consistent representation
        german_hash = hashlib.md5(example['de'].encode()).hexdigest()
        # Convert hash to seed for reproducible random generation
        seed = int(german_hash[:8], 16)
        torch.manual_seed(seed)
        target_representation = torch.randn(768)  # Deterministic based on German text
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'task_name': 'translation',
            'task_id': 4,
            'target': target_representation,  # [768] deterministic representation
            'target_text': example['de'],  # Keep original for reference
            'input_text': example['en']
        }


class DatasetLoader:
    """
    Main dataset loader for DYNAMO.
    Handles loading and preprocessing of all task datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Handle both Config objects and dictionaries
        if hasattr(config, 'data'):
            # Config object
            self.data_config = config.data
        else:
            # Dictionary
            self.data_config = config.get("data", {})
        
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Dataset sizes
        self.dataset_sizes = {
            'sentiment': getattr(self.data_config, 'sst2_size', 10000),
            'qa': getattr(self.data_config, 'squad_size', 20000),
            'summarization': getattr(self.data_config, 'xsum_size', 15000),
            'code_generation': getattr(self.data_config, 'code_gen_size', 8000),
            'translation': getattr(self.data_config, 'translation_size', 12000)
        }
        
        self.max_length = getattr(self.data_config, 'max_input_length', 512)
        self.cache_dir = getattr(self.data_config, 'cache_dir', './cache')
        
        # Real datasets configuration  
        self.real_datasets = {
            'sentiment': ('sst2', None),  # Stanford Sentiment Treebank
            'qa': ('squad', None),  # SQuAD Question Answering
            'summarization': ('cnn_dailymail', '3.0.0'),  # CNN/DailyMail (instead of XSum)
            'code_generation': ('mbpp', None),  # Mostly Basic Python Problems (instead of CodeSearchNet)
            'translation': ('wmt14', 'de-en')  # WMT14 German-English
        }
        
        logger.info("Dataset loader initialized")
    
    def load_sentiment_data(self, split: str = 'train') -> List[Dict]:
        """Load SST-2 sentiment analysis data."""
        logger.info(f"Loading SST-2 {split} data...")
        
        try:
            dataset = load_dataset('sst2', split=split, cache_dir=self.cache_dir)
            
            # Convert to our format
            data = []
            max_size = self.dataset_sizes['sentiment']
            
            for i, example in enumerate(dataset):
                if i >= max_size:
                    break
                
                data.append({
                    'sentence': example['sentence'],
                    'label': example['label']
                })
            
            logger.info(f"Loaded {len(data)} SST-2 examples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load SST-2: {e}. Using synthetic data.")
            return self._create_synthetic_sentiment_data()
    
    def load_qa_data(self, split: str = 'train') -> List[Dict]:
        """Load SQuAD question answering data."""
        logger.info(f"Loading SQuAD {split} data...")
        
        try:
            dataset = load_dataset('squad', split=split, cache_dir=self.cache_dir)
            
            data = []
            max_size = self.dataset_sizes['qa']
            
            for i, example in enumerate(dataset):
                if i >= max_size:
                    break
                
                # Get first answer (SQuAD can have multiple answers)
                answer = example['answers']['text'][0] if example['answers']['text'] else ""
                start_pos = example['answers']['answer_start'][0] if example['answers']['answer_start'] else 0
                
                data.append({
                    'question': example['question'],
                    'context': example['context'],
                    'answer': answer,
                    'start_position': min(start_pos, 510),  # Clamp to max length
                    'end_position': min(start_pos + len(answer), 511)
                })
            
            logger.info(f"Loaded {len(data)} SQuAD examples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}. Using synthetic data.")
            return self._create_synthetic_qa_data()
    
    def load_summarization_data(self, split: str = 'train') -> List[Dict[str, str]]:
        """Load text summarization data (CNN/DailyMail)."""
        try:
            logger.info("Loading CNN/DailyMail summarization dataset...")
            dataset = load_dataset('cnn_dailymail', '3.0.0', split=split, cache_dir=self.cache_dir)
            
            # Sample if dataset is too large
            if len(dataset) > self.dataset_sizes['summarization']:
                dataset = dataset.shuffle(seed=42).select(range(self.dataset_sizes['summarization']))
            
            data = []
            for item in dataset:
                # CNN/DailyMail format: article -> highlights
                data.append({
                    'document': item['article'],
                    'summary': item['highlights'],
                    'task_type': 'summarization'
                })
            
            logger.info(f"Loaded {len(data)} CNN/DailyMail examples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load CNN/DailyMail dataset: {e}")
            return self._generate_synthetic_summarization_data()
    
    def load_code_generation_data(self, split: str = 'train') -> List[Dict[str, str]]:
        """Load code generation data (MBPP - Mostly Basic Python Problems)."""
        try:
            logger.info("Loading MBPP code generation dataset...")
            dataset = load_dataset('mbpp', split=split, cache_dir=self.cache_dir)
            
            # Sample if dataset is too large
            if len(dataset) > self.dataset_sizes['code_generation']:
                dataset = dataset.shuffle(seed=42).select(range(self.dataset_sizes['code_generation']))
            
            data = []
            for item in dataset:
                # MBPP format: text description -> code
                data.append({
                    'text': item['text'],  # Problem description
                    'code': item['code'],  # Python code solution
                    'task_type': 'code_generation'
                })
            
            logger.info(f"Loaded {len(data)} MBPP examples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load MBPP dataset: {e}")
            return self._generate_synthetic_code_generation_data()
    
    def load_translation_data(self, split: str = 'train') -> List[Dict]:
        """Load translation data."""
        logger.info(f"Loading translation {split} data...")
        
        try:
            # Try to load WMT or similar dataset
            dataset = load_dataset('wmt14', 'de-en', split=split, cache_dir=self.cache_dir)
            
            data = []
            max_size = self.dataset_sizes['translation']
            
            for i, example in enumerate(dataset):
                if i >= max_size:
                    break
                
                # Fix field names to match TranslationDataset expectations
                data.append({
                    'en': example['translation']['en'],  # English text
                    'de': example['translation']['de']   # German text
                })
            
            logger.info(f"Loaded {len(data)} translation examples")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load translation data: {e}. Using synthetic data.")
            return self._create_synthetic_translation_data()
    
    def _create_synthetic_sentiment_data(self) -> List[Dict]:
        """Create synthetic sentiment data for testing."""
        data = []
        positive_examples = [
            "This movie is amazing and wonderful!",
            "I love this product, it's fantastic.",
            "Great service and excellent quality.",
            "Outstanding performance and brilliant acting.",
            "Absolutely perfect and highly recommended."
        ]
        negative_examples = [
            "This movie is terrible and boring.",
            "I hate this product, it's awful.",
            "Poor service and bad quality.",
            "Disappointing performance and bad acting.",
            "Completely useless and not recommended."
        ]
        
        for i in range(self.dataset_sizes['sentiment']):
            if i % 2 == 0:
                sentence = random.choice(positive_examples)
                label = 1
            else:
                sentence = random.choice(negative_examples)
                label = 0
            
            data.append({'sentence': sentence, 'label': label})
        
        return data
    
    def _create_synthetic_qa_data(self) -> List[Dict]:
        """Create synthetic QA data for testing."""
        data = []
        templates = [
            {
                'context': "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                'question': "What is the capital of France?",
                'answer': "Paris",
                'start_position': 23,
                'end_position': 28
            },
            {
                'context': "Python is a programming language. It was created by Guido van Rossum.",
                'question': "Who created Python?",
                'answer': "Guido van Rossum",
                'start_position': 53,
                'end_position': 69
            }
        ]
        
        for i in range(self.dataset_sizes['qa']):
            template = templates[i % len(templates)]
            data.append(template.copy())
        
        return data
    
    def _generate_synthetic_summarization_data(self) -> List[Dict]:
        """Create synthetic summarization data for testing."""
        data = []
        examples = [
            {
                'document': "Artificial intelligence is transforming many industries. Machine learning algorithms can process vast amounts of data and identify patterns that humans might miss. This technology is being used in healthcare, finance, and transportation.",
                'summary': "AI and machine learning are transforming industries like healthcare, finance, and transportation."
            },
            {
                'document': "Climate change is one of the most pressing issues of our time. Rising temperatures, melting ice caps, and extreme weather events are affecting ecosystems worldwide. Governments and organizations are working to reduce carbon emissions.",
                'summary': "Climate change causes rising temperatures and extreme weather, prompting efforts to reduce emissions."
            }
        ]
        
        for i in range(self.dataset_sizes['summarization']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def _generate_synthetic_code_generation_data(self) -> List[Dict]:
        """Create synthetic code generation data for testing."""
        data = []
        examples = [
            {
                'text': "Function to calculate the factorial of a number",
                'code': "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
            },
            {
                'text': "Function to check if a number is prime",
                'code': "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
            }
        ]
        
        for i in range(self.dataset_sizes['code_generation']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def _create_synthetic_translation_data(self) -> List[Dict]:
        """Create synthetic translation data for testing."""
        data = []
        examples = [
            {'en': "Hello, how are you?", 'de': "Hallo, wie geht es dir?"},
            {'en': "I love music.", 'de': "Ich liebe Musik."},
            {'en': "The weather is nice today.", 'de': "Das Wetter ist schön heute."},
            {'en': "Where is the library?", 'de': "Wo ist die Bibliothek?"}
        ]
        
        for i in range(self.dataset_sizes['translation']):
            example = examples[i % len(examples)]
            data.append(example.copy())
        
        return data
    
    def create_datasets(self, split: str = 'train') -> Dict[str, Dataset]:
        """
        Create all task datasets.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
        
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        
        # Load data for each task
        sentiment_data = self.load_sentiment_data(split)
        qa_data = self.load_qa_data(split)
        summarization_data = self.load_summarization_data(split)
        code_data = self.load_code_generation_data(split)
        translation_data = self.load_translation_data(split)
        
        # Create datasets
        datasets['sentiment'] = SentimentDataset(
            sentiment_data, self.tokenizer, self.max_length, 'sentiment'
        )
        datasets['qa'] = QADataset(
            qa_data, self.tokenizer, self.max_length, 'qa'
        )
        datasets['summarization'] = SummarizationDataset(
            summarization_data, self.tokenizer, self.max_length, 'summarization'
        )
        datasets['code_generation'] = CodeGenerationDataset(
            code_data, self.tokenizer, self.max_length, 'code_generation'
        )
        datasets['translation'] = TranslationDataset(
            translation_data, self.tokenizer, self.max_length, 'translation'
        )
        
        logger.info(f"Created {len(datasets)} task datasets for {split} split")
        return datasets
    
    def create_dataloaders(
        self,
        datasets: Dict[str, Dataset],
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all datasets.
        
        Args:
            datasets: Dictionary of datasets
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        
        for task_name, dataset in datasets.items():
            dataloaders[task_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        return dataloaders

