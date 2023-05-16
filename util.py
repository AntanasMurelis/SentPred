import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

def split_sentiment(data):
    # Split the 'Sentiment' column into two separate columns
    data[['Positive_Sentiment', 'Negative_Sentiment']] = data['Sentiment'].str.split(' ', expand=True)

    # Convert the sentiment columns to integers
    data['Positive_Sentiment'] = data['Positive_Sentiment'].astype(int)
    data['Negative_Sentiment'] = data['Negative_Sentiment'].astype(int)

    # Calculate the average sentiment and create a new column
    data['Average_Sentiment'] = (data['Positive_Sentiment'] + data['Negative_Sentiment']) / 2

    # Create a new column 'Sentiment_Class' based on the 'Average_Sentiment' column
    data['Sentiment_Class'] = data['Average_Sentiment'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))

    # Drop the original 'Sentiment' column
    data.drop(columns=['Sentiment'], inplace=True)

    return data

def split_dataframe(df, train_size=0.85, val_size=0.05, test_size=0.1, random_state=None):

    # Split the DataFrame into training and test sets
    train_df, test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=random_state)

    # Split the training set into training and validation sets
    train_df, val_df = train_test_split(train_df, train_size=train_size/(train_size+val_size), test_size=val_size/(train_size+val_size), random_state=random_state,
                                        shuffle = True)

    return train_df, val_df, test_df


def stratified_sample(df, sample_size, label_col):
    if isinstance(sample_size, float):
        # When sample_size is a fraction
        return df.groupby(label_col).apply(lambda x: x.sample(frac=sample_size, random_state=42)).reset_index(drop=True)
    elif isinstance(sample_size, int):
        # When sample_size is an absolute number
        return df.groupby(label_col).apply(lambda x: x.sample(n=min(sample_size, len(x)), random_state=42)).reset_index(drop=True)
    else:
        raise ValueError("sample_size should either be a float (fraction) or int (absolute number)")
    
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


def create_performance_table(save_dir, n_unfrozen_layers_list):
    # Define the metrics you want to collect
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    results = []

    for n_unfrozen_layers in n_unfrozen_layers_list:
        metrics_file = os.path.join(save_dir, f"checkpoints_single_screen_layers_{n_unfrozen_layers}", "eval_metrics.json")
        
        # Check if the file exists
        if not os.path.isfile(metrics_file):
            print(f"File not found: {metrics_file}")
            continue

        # Load the JSON file
        with open(metrics_file, 'r') as f:
            eval_metrics = json.load(f)

        # Collect the metrics
        result = [n_unfrozen_layers]
        for metric in metrics:
            result.append(eval_metrics.get(metric, None))
        results.append(result)

    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['Unfrozen Layers'] + metrics)

    return df