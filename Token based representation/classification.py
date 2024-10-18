from gensim.models import Word2Vec
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from AST_classification import strip_comments
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time

from tokenizers.homemade_tokenizerv2 import get_string, normalise_identifiers, tokenize_code
from tokenizers.homemade_tokenizer import main as old_tokenizer


# nltk.download('punkt_tab')


# Defining struct for dataset

class TextDataset(Dataset):
    def __init__(self, programs_dict, tokenizer):
        self.texts = []
        self.labels = []
        self.max_len = 0
        self.tokenizer = tokenizer
        for key in programs_dict.keys():
            for ls in programs_dict[key]:
                if len(ls) > self.max_len: self.max_len = len(ls)
            self.texts += programs_dict[key]
            self.labels += [key] * len(programs_dict[key])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize and pad sequences
        sequence = self.tokenizer.texts_to_sequences([self.texts[idx]])[0]
        sequence = pad_sequences([sequence], maxlen=self.max_len, padding='post')[0]
        label = self.labels[idx]  # Label as an integer
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Defining the LSTM model

class LSTMModel(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        # freeze=True makes it so that the embeddings are NOT updated during training
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, classes+1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x



def create_program_dict(folders_amount: int, items_per_folder: int, rm_comments, normalise):
    programs_dict = {}
    limit = items_per_folder
    classes = folders_amount
    for i in range(classes):
        folder_path = 'Thesis Data/ProgramData/' + str(i)
        files = glob.glob(os.path.join(folder_path, '*'))
        count = 0
        for file in files:
            with open(file, 'r') as f:
                src_code = f.read()
                if rm_comments:
                    src_code = strip_comments(src_code)
                if normalise:
                    src_code = get_string(normalise_identifiers(tokenize_code(src_code)))
                if i not in programs_dict.keys():
                    programs_dict[i] = [src_code]
                else:
                    programs_dict.get(i).append(src_code)
            count += 1
            if count > + limit: break
    return programs_dict

def create_embedding_matrix(word2vecmodel, embedding_dim, word_index):
    word_vectors = word2vecmodel.wv
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            # Randomly initialize the embedding for out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

    return embedding_matrix

def program(classes, items_per_folder, epochs, rm_comments, normalise):
    total_predictions = []
    total_labels = []
    for i in range(epochs):
        predictions, labels = iteration(classes, items_per_folder, rm_comments, normalise)
        total_predictions += predictions
        total_labels += labels
    return total_predictions, total_labels

def iteration(classes, items_per_folder, rm_comments, normalise):
    classes = classes
    items_per_folder = items_per_folder
    embedding_dim = 100
    programs_dict = create_program_dict(folders_amount=classes+1, items_per_folder=items_per_folder, rm_comments=rm_comments, normalise=normalise)

    # Tokenize the corpus
    tokenized_corpus_list = [[old_tokenizer(item, False, False) for item in ls] for ls in programs_dict.values()]
    tokenized_corpus = []
    for inner in tokenized_corpus_list: tokenized_corpus += inner

    # Train Word2Vec model on the compiled corpus
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=25, min_count=1, workers=4, epochs=10)
    # Save the model
    # word2vec_model.save("word2vec.model")

    # Create tokenizer that is useful for the word_index and accessing the w2v vectors later
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_corpus)
    sequences = tokenizer.texts_to_sequences(tokenized_corpus)
    word_index = tokenizer.word_index
    # Loading into dataloader class using the same tokenizer to untokenize the sentences

    dataset = TextDataset(programs_dict=programs_dict, tokenizer=tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


    # iwanne be
    # PERFECT
    # I KNOW ILL neva be
    # perfect

    # IF YOU COULD CHOOSE ONLY 1 THING TO FIX
    # worthless...


    embedding_matrix = create_embedding_matrix(word2vec_model, embedding_dim, word_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(embedding_matrix, embedding_dim, classes).to(device)

    criterion = nn.CrossEntropyLoss()  # For multi class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    # TRAIN
    for epoch in range(num_epochs):
        timer = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        count = 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            outputs = model(inputs)  # Model prediction
            loss = criterion(outputs, labels)  # Compute loss

            # Backpropagation
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters
            running_loss += loss.item() * inputs.size(0)
        print(f'Time taken: {time.time() - timer}')

        epoch_loss = running_loss / len(train_loader.dataset)  # Average loss per epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')


    # TESTING

    # Store predictions and actual labels for the whole test set
    all_preds = []
    all_labels = []

    # Turn off gradients for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Generate classification report
    # print(classification_report(all_labels, all_preds))

    return all_preds, all_labels




if __name__ == '__main__':
    print('Starting 1')
    raw_code_predictions, raw_code_labels = program(classes=50, items_per_folder=100, epochs=3, rm_comments=False, normalise=False)
    print('Starting 2')
    no_comments_predictions, no_comments_labels = program(classes=50, items_per_folder=100, epochs=3, rm_comments=True, normalise=False)
    print('Starting 3')
    normalised_predictions, normalised_labels = program(classes=50, items_per_folder=100, epochs=3, rm_comments=True, normalise=True)

    accuracy_raw = accuracy_score(raw_code_predictions, raw_code_labels)
    precision_raw = precision_score(raw_code_predictions, raw_code_labels, average='macro')
    recall_raw = recall_score(raw_code_predictions, raw_code_labels, average='macro')
    f1_raw = f1_score(raw_code_predictions, raw_code_labels, average='macro')

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
    scores_raw = [accuracy_raw, precision_raw, recall_raw, f1_raw]
    scores_nc = [accuracy_nc, precision_nc, recall_nc, f1_nc]
    scores_norm = [accuracy_norm, precision_norm, recall_norm, f1_norm]

    all_scores = [scores_raw, scores_nc, scores_norm]
    conditions = ['Raw Code', 'No Comments', 'Normalised']  # Names for each plot
    colors = ['green', 'blue', 'red'] # Green for raw, blue for nc, red for norm

    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns of plots

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
        ax.bar(conditions, [scores_raw[i], scores_nc[i], scores_norm[i]], color=colors)
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
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Normalised Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(no_comments_predictions, no_comments_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('No comments')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(raw_code_predictions, raw_code_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Raw Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()