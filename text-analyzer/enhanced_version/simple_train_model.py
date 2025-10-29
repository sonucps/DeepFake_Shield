import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SimpleModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        print("🚀 Initializing Simple Model Trainer...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"✅ Using device: {self.device}")
    
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        print("📊 Starting training process...")
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if batch_idx % 2 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"📍 Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
            
            # Validate
            accuracy = self.validate(val_loader)
            print(f"📍 Validation Accuracy: {accuracy:.2f}%")
    
    def validate(self, val_loader):
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        accuracy = accuracy_score(actual_labels, predictions) * 100
        self.model.train()
        return accuracy
    
    def save_model(self, output_dir="./trained_model"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ Model saved to {output_dir}")

def main():
    try:
        # Load data from CSV
        print("📁 Loading training data...")
        if not os.path.exists('./training_data.csv'):
            print("❌ training_data.csv not found!")
            print("Please run 'python create_simple_data.py' first")
            return
        
        df = pd.read_csv('./training_data.csv')
        print(f"✅ Loaded {len(df)} samples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Initialize trainer
        trainer = SimpleModelTrainer()
        
        # Create datasets
        train_dataset = TextDataset(
            texts=train_df['text'].values,
            labels=train_df['label'].values,
            tokenizer=trainer.tokenizer
        )
        
        val_dataset = TextDataset(
            texts=val_df['text'].values,
            labels=val_df['label'].values, 
            tokenizer=trainer.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Train model
        trainer.train(train_loader, val_loader, epochs=3)
        
        # Save model
        trainer.save_model()
        
        print("\n🎉 Training completed successfully!")
        print("Model saved to './trained_model' folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have installed:")
        print("pip install transformers torch pandas scikit-learn")

if __name__ == "__main__":
    main()