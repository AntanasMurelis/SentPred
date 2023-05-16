import os
from evaluations import evaluate_metrics, evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd

def train(model, data_loader, device, lr, num_epochs, record_every=100, save_every=1000, save_dir='checkpoints'):
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    total_steps = len(data_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.5*total_steps, num_training_steps=total_steps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Store losses for each epoch
    train_losses = []

    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Record train loss every 'record_every' iterations
            if iteration % record_every == 0:
                train_losses.append(loss.item())
                print(f"Iteration {iteration}, Train Loss: {loss.item():.4f}")
            
            # Save model and dataloader every 'save_every' iterations
            if iteration % save_every == 0 or iteration % (total_steps-1) == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}_iter{iteration}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                    'data_loader': data_loader
                }, checkpoint_path)
                print(f"Saved checkpoint at: {checkpoint_path}")
            
            iteration += 1

    return train_losses


def load_model(checkpoint_path, model, tokenizer, optimizer=None, scheduler=None):
    # Check if the checkpoint exists
    if os.path.exists(checkpoint_path):
        # Define device to load on
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load the model, optimizer, and scheduler states
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Get the epoch and iteration from the checkpoint
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        
        print(f"Checkpoint loaded from {checkpoint_path}. Starting from epoch {start_epoch + 1}, iteration {start_iteration + 1}.")
        return model, tokenizer, optimizer, scheduler, start_epoch, start_iteration
    else:
        print("No checkpoint found. Starting from scratch.")
        return model, tokenizer, optimizer, scheduler, 0, 0
    
def freeze_bert_layers(model, n_unfrozen_layers):
    total_layers = len(list(model.base_model.encoder.layer))
    freeze_layers = total_layers - n_unfrozen_layers

    # Freeze the specified number of layers
    for idx, layer in enumerate(model.base_model.encoder.layer):
        if idx < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

def freeze_bert_layers_except(model, n_unfrozen_layer):
    total_layers = len(list(model.base_model.encoder.layer))

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the specified layer
    for param in model.base_model.encoder.layer[total_layers - n_unfrozen_layer - 1].parameters():
        param.requires_grad = True


import json

import seaborn as sns

def search_best_unfrozen_layers(train_data_loader, val_data_loader, device, lr, num_epochs, n_unfrozen_layers_list, save_every, save_dir, freeze_type='multiple', record_every=100):
    best_acc = 0
    best_unfrozen_layers = 0
    losses_dfs = []

    for n_unfrozen_layers in n_unfrozen_layers_list:
        print(f"Training with {n_unfrozen_layers} unfrozen layers")

        # Load a fresh model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

        # Freeze layers based on the freeze type
        if freeze_type == 'multiple':
            freeze_bert_layers(model, n_unfrozen_layers)
        elif freeze_type == 'single':
            freeze_bert_layers_except(model, n_unfrozen_layers)
        else:
            raise ValueError(f"Invalid freeze_type: {freeze_type}. Expected 'multiple' or 'single'.")

        model.to(device)

        # Train the model with the specified number of unfrozen layers and get the losses
        train_losses = train(model = model, data_loader = train_data_loader, device = device, lr = lr, num_epochs = num_epochs, 
                             save_every = save_every, record_every = record_every, save_dir = f"{save_dir}_layers_{n_unfrozen_layers}")

        # Create a DataFrame from the training losses
        losses_df = pd.DataFrame({
            'Step': range(len(train_losses)),
            'Train Loss': train_losses,
            'Layers': [n_unfrozen_layers] * len(train_losses)
        })

        losses_dfs.append(losses_df)

        # Save the training losses as JSON files
        with open(f"{save_dir}_layers_{n_unfrozen_layers}/train_losses.json", "w") as f:
            json.dump(train_losses, f)

        # Evaluate the model on the validation dataset
        eval_metrics = evaluate_metrics(model, val_data_loader, device)
        acc = eval_metrics['accuracy']

        # Save the evaluation metrics as a JSON file
        with open(f"{save_dir}_layers_{n_unfrozen_layers}/eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f)

        print(f"Accuracy with {n_unfrozen_layers} unfrozen layers: {acc:.4f}")

        # Update the best unfrozen layers and accuracy if the current configuration is better
        if acc > best_acc:
            best_acc = acc
            best_unfrozen_layers = n_unfrozen_layers

    # Combine all loss DataFrames
    all_losses_df = pd.concat(losses_dfs)

    # Melt the DataFrame to make it suitable for seaborn
    all_losses_df = pd.melt(all_losses_df, id_vars=['Step', 'Layers'], var_name='Loss Type', value_name='Loss')

    sns.set_palette("pastel")

    # Plot the training losses with different number of unfrozen layers using seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=all_losses_df, x='Step', y='Loss', hue='Layers')
    plt.title('Train Losses with Different Numbers of Unfrozen Layers')
    plt.show()

    return best_unfrozen_layers, best_acc


