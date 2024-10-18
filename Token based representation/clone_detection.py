from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tokenizers.homemade_tokenizer_java import main_java as tokenize_java
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import time
import pandas as pd
import random

class TextDataset(Dataset):
    def __init__(self, clones_list, false_pos_list, pretty_printed, max_len, tokenizer):
        self.inputs = []
        self.labels = []
        self.tokenized_dict = pretty_printed
        self.max_len = max_len
        self.tokenizer = tokenizer
        for clone in clones_list:
            self.inputs.append(clone)
            self.labels.append(0)
        for false_pos in false_pos_list:
            self.inputs.append(false_pos)
            self.labels.append(1)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Tokenize and pad sequences
        sequence1 = self.tokenizer.texts_to_sequences([self.tokenized_dict[self.inputs[idx][0]]])[0]
        sequence1 = pad_sequences([sequence1], maxlen=self.max_len, padding='post')[0]
        sequence2 = self.tokenizer.texts_to_sequences([self.tokenized_dict[self.inputs[idx][1]]])[0]
        sequence2 = pad_sequences([sequence2], maxlen=self.max_len, padding='post')[0]
        label = self.labels[idx]  # Label as an integer
        return torch.tensor(sequence1, dtype=torch.long), torch.tensor(sequence2, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)  # Binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward_once(self, x):
        embedded = self.embedding(x)
        # Encode one input
        output, (hn, cn) = self.lstm(embedded)
        return hn[-1]  # Return the last hidden state

    def forward(self, input1, input2):
        # Forward both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Compute distance (e.g., cosine similarity or difference)
        distance = torch.abs(output1 - output2)
        # Pass through a fully connected layer for binary classification
        out = self.fc(distance)
        return torch.sigmoid(out)


def iteration(samples, clone_percentage, rm_comments, normalise):

    pretty_printed = pd.read_csv('../Thesis Data/BCB/pretty_printed_functions.csv', encoding='utf-8', index_col='function_id')
    clones = pd.read_csv('../Thesis Data/BCB/clones.csv', encoding='utf-8')
    false_positives = pd.read_csv('../Thesis Data/BCB/false_positives.csv', encoding='utf-8')

    functions = set()

    # Samples from each 'clones' and 'false_positives'
    clones_count = int(clone_percentage * samples)
    fp_count = samples - clones_count

    count = 0
    clones_list = []
    var = list(zip(list(clones['function_id_one']), list(clones['function_id_two'])))
    clones_list = random.choices(var, k=clones_count)

    var = list(zip(list(false_positives['function_id_one']), list(false_positives['function_id_two'])))
    false_pos_list = random.choices(var, k=fp_count)
    for id1,id2 in clones_list:
        functions.add(id1)
        functions.add(id2)

    for id1,id2 in false_pos_list:
        functions.add(id1)
        functions.add(id2)


    tokenized_corpus = []
    tokenized_dict = {}
    max_len = 0
    for function in functions:
        temp = tokenize_java(pretty_printed.at[function, 'text'], rm_comments, normalise)
        max_len = max(max_len, len(temp))
        tokenized_corpus.append(temp)
        tokenized_dict[function] = temp


    # embedding_dim is the size of the vector
    embedding_dim = 150

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=30, min_count=1, workers=4, epochs=10)
    # Save the model
    word2vec_model.save("word2vec_java.model")

    # setting up the tokenizer to store each word as an index
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_corpus)
    sequences = tokenizer.texts_to_sequences(tokenized_corpus)
    word_index = tokenizer.word_index

    # Setting up the dataset
    dataset = TextDataset(clones_list=clones_list, false_pos_list=false_pos_list, pretty_printed=tokenized_dict, max_len=max_len, tokenizer=tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Creating the embedding matrix to access word2vec vectors from token of word
    word_vectors = word2vec_model.wv
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            # Randomly initialize the embedding for out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))


    # iwanne be
    # PERFECT
    # I KNOW ILL neva be
    # perfect

    # IF YOU COULD CHOOSE ONLY 1 THING TO FIX
    # worthless...




    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(embedding_matrix, embedding_dim).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5  # Number of epochs to train

    print('Starting Training...')
    # Training loop
    for epoch in range(num_epochs):
        timer = time.time()
        model.train()  # Set the model in training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for code1, code2, labels in train_loader:
            # Move data to GPU if available
            code1, code2, labels = code1.to(device), code2.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass: get predictions from the model
            outputs = model(code1, code2)  # Outputs are between 0 and 1 (sigmoid)

            # Compute loss
            loss = criterion(outputs.squeeze(), labels.float())  # Labels need to be float for BCELoss

            # Backward pass: compute gradients
            loss.backward()
            optimizer.step()  # Update model parameters

            # Track the loss and accuracy
            running_loss += loss.item() * labels.size(0)  # Accumulate loss

            # Convert predictions to binary (clone or non-clone)
            predicted = (outputs.squeeze() > 0.5).float()  # Squeeze the outputs to match labels

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total

        # Print epoch results
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        print(f'Time taken: {time.time() - timer}')
    # Evaluate the model on test data
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for code1, code2, labels in test_loader:
            code1, code2, labels = code1.to(device), code2.to(device), labels.to(device)
            outputs = model(code1, code2)
            predicted = (outputs.squeeze() > 0.5).float()  # Threshold at 0.5 for binary classification
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return all_preds, all_labels

    # Optional: Detailed performance metrics
    # print(classification_report(all_labels, all_preds))

def program(epochs, samples, clone_percentage, rm_comments, normalise):
    total_predictions = []
    total_labels = []
    for i in range(epochs):
        predictions, labels = iteration(samples, clone_percentage, rm_comments, normalise)
        total_predictions += predictions
        total_labels += labels
    return total_predictions, total_labels

if __name__ == '__main__':
    print('Starting 1')
    # raw_code_predictions, raw_code_labels = program(2, 500, 0.75, False, False)
    print('Starting 2')
    no_comments_predictions, no_comments_labels = program(2, 650, 0.75, True, False)
    print('Starting 3')
    normalised_predictions, normalised_labels = program(2, 820, 0.75, True, True)

    """accuracy_raw = accuracy_score(raw_code_predictions, raw_code_labels)
    precision_raw = precision_score(raw_code_predictions, raw_code_labels, average='macro')
    recall_raw = recall_score(raw_code_predictions, raw_code_labels, average='macro')
    f1_raw = f1_score(raw_code_predictions, raw_code_labels, average='macro')"""

    accuracy_nc = accuracy_score(no_comments_predictions, no_comments_labels)
    precision_nc = precision_score(no_comments_predictions, no_comments_labels, average='macro')
    recall_nc = recall_score(no_comments_predictions, no_comments_labels, average='macro')
    f1_nc = f1_score(no_comments_predictions, no_comments_labels, average='macro')

    accuracy_norm = accuracy_score(normalised_predictions, normalised_labels)
    precision_norm = precision_score(normalised_predictions, normalised_labels, average='macro')
    recall_norm = recall_score(normalised_predictions, normalised_labels, average='macro')
    f1_norm = f1_score(normalised_predictions, normalised_labels, average='macro')

    # Metrics and their corresponding names
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # scores_raw = [accuracy_raw, precision_raw, recall_raw, f1_raw]
    scores_nc = [accuracy_nc, precision_nc, recall_nc, f1_nc]
    scores_norm = [accuracy_norm, precision_norm, recall_norm, f1_norm]

    all_scores = [scores_nc, scores_norm]
    conditions = ['No Comments', 'Normalised']  # Names for each plot
    colors = ['blue', 'red']  # Green for raw, blue for nc, red for norm

    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 3 columns of plots

    # Loop through each condition and corresponding scores to plot them
    for i, ax in enumerate(axes):
        ax.bar(metrics, all_scores[i], color=colors[i])
        print(all_scores[i])
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(conditions[i], fontsize=16)
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns for 4 metrics

    # Loop through each metric and plot raw, nc, and norm as bars with the specified colors
    for i, ax in enumerate(axes2):
        ax.bar(conditions, [scores_nc[i], scores_norm[i]], color=colors)
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(metrics[i], fontsize=16)
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout for the second set of plots
    plt.tight_layout()

    # Show both sets of plots
    plt.show()

    # Confustion Matrix
    cm = confusion_matrix(normalised_predictions, normalised_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Normalised Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(no_comments_predictions, no_comments_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('No comments')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    """cm = confusion_matrix(raw_code_predictions, raw_code_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Raw Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()"""