import torch
from transformers import BertForSequenceClassification, BertTokenizer
from utils import load_model
import pickle

def get_last_layer_representations(df, model_path, checkpoint_path, layers, column_name='text'):
    # Define device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the pretrained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model = model.to(device)  # Move model to device
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load the checkpoint if it exists
    model, _, _, _, _, _ = load_model(checkpoint_path, model, tokenizer)

    # Ensure model is in evaluation mode
    model.eval()

    # Initialize a list to hold the representations
    representations = []
    sentiment_labels = []

    for i, row in df.iterrows():
        # Tokenize the text and convert to tensors
        inputs = tokenizer(row[column_name], return_tensors='pt').to(device)  # Move inputs to device

        # Get the output from the model
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]

        # Get the representation for the [CLS] token (first token)
        cls_representation = last_hidden_state[0, 0, :].cpu().numpy()  # Move tensor to cpu before converting to numpy

        # Append the representation to the list
        representations.append(cls_representation)
        sentiment_labels.append(row['Sentiment_Class'])  # Assuming 'Sentiment_Class' is the name of your sentiment label column

    # Convert to DataFrame
    labeled_representations = pd.DataFrame({
        'embeddings': representations,
        'sentiment_labels': sentiment_labels
    })

    # Save to a pickle file
    with open(f'labeled_representations_single_{layers}.pkl', 'wb') as f:
        pickle.dump(labeled_representations, f)

    return representations