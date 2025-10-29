from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
import json

class ModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        print("🚀 Initializing Model Trainer...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3
        )
        print("✅ Model and tokenizer loaded successfully!")
        
    def tokenize_function(self, examples):
        """Tokenize the texts"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors=None
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, dataset, output_dir="./ai_detection_model"):
        """Train the model"""
        print("📊 Starting training process...")
        
        # Tokenize dataset
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='./logs',
            logging_steps=5,
            warmup_steps=50,
            report_to=None,
            save_total_limit=2
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=128
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("🎯 Starting model training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to {output_dir}")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        return trainer

def load_json_dataset(json_path):
    """Load dataset from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to datasets format
    dataset_dict = {
        'text': data['texts'],
        'label': data['labels']
    }
    
    return Dataset.from_dict(dataset_dict)

# Main execution
if __name__ == "__main__":
    try:
        # Try multiple possible paths
        possible_paths = [
            "./training_data.json",
            "training_data.json", 
            "../training_data.json",
            "./enhanced_version/training_data.json"
        ]
        
        data_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 Found training data at: {path}")
                data_path = path
                break
        
        if data_path is None:
            print("❌ Training data not found!")
            print("Please run 'python create_data.py' first to create the training data.")
            print("Looking for: training_data.json")
            exit(1)
            
        # Load dataset from JSON
        print(f"📁 Loading training dataset from: {data_path}")
        dataset = load_json_dataset(data_path)
        print(f"✅ Loaded dataset with {len(dataset)} samples")
        
        # Train model
        trainer = ModelTrainer()
        trainer.train(dataset)
        
        print("\n🎉 Training completed successfully!")
        print("You can now use the trained model for AI text detection!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\nTroubleshooting steps:")
        print("1. Run 'python create_data.py' first")
        print("2. Make sure 'training_data.json' exists")
        print("3. Check if all packages are installed:")
        print("   pip install transformers datasets torch scikit-learn")