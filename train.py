from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import PainterDataset, PainterModel
import warnings
# warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    losses = []
    correct_predictions = 0
    
    pbar = tqdm(data_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        # Clip gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        _, preds = torch.max(outputs, dim=1) # ??
        correct_predictions += torch.sum(preds == labels)
        current_mean_loss = np.mean(losses)
        current_acc = (correct_predictions.double() / ((pbar.n + 1) * data_loader.batch_size)).item()
        pbar.set_postfix({
                'loss': f'{current_mean_loss:.4f}',
                'acc': f'{current_acc:.4f}'
        })
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), losses

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), predictions, true_labels)

def main():
    MODEL_NAME = 'distilbert-base-uncased'  # Small, efficient model
    MAX_LENGTH = 128
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    DP_RATE = 0.3
    SEED = 42

    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = PainterDataset("data/train.csv", tokenizer, MAX_LENGTH)
    val_dataset = PainterDataset("data/test.csv", tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = PainterModel(MODEL_NAME, n_classes=2, dropout=DP_RATE)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    best_accuracy = 0
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss, train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler
        )
        
        val_acc, val_loss, _, _ = eval_model(model, val_loader, loss_fn, device)
        
        for loss in train_losses:
            history['train_acc'].append(None)
            history['train_loss'].append(loss)
            history['val_acc'].append(None)
            history['val_loss'].append(None)

        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc.item())
        history['val_loss'].append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    val_acc, val_loss, predictions, true_labels = eval_model(model, val_loader, loss_fn, device)
    
    print(f'\nFinal Validation Results:')
    print(f'Accuracy: {val_acc:.4f}')
    print(f'Loss: {val_loss:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(true_labels, predictions))
    
    return model, tokenizer, history

if __name__ == "__main__":
    model, tokenizer, history = main()
    log = pd.DataFrame(history)
    log[log.val_loss.isna()].train_loss.rolling(10, center=True).mean().plot(title='Smoothed training loss over learning steps')
    plt.show()
    log.to_csv('training_log.csv')