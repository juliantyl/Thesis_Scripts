import re
import glob
import os
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from gensim.models import Word2Vec
from pycparser import c_parser
from CFGFile_C import CFGBuilder
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tokenizers.homemade_tokenizerv2 import tokenize_code, normalise_identifiers, get_string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


class GraphDataset(Dataset):
    def __init__(self, graph_dict, tokenizer):
        self.graphs = []
        self.labels = []
        self.max_len = 0
        self.tokenizer = tokenizer
        for label, graphs in graph_dict.items():
            self.graphs += graphs
            for graph in graphs:
                for node in graph.nodes():
                    if len(node.tokens) > self.max_len:
                        self.max_len = len(node.tokens)
            self.labels += [label] * len(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        cfg = copy.deepcopy(self.graphs[idx])
        label = self.labels[idx]
        cfg = self.tokenize_cfg(cfg)

        # Aggregate data into usable form by getting node features and edge index
        node_indices = list(cfg.nodes())
        node_id_map = {node_id: idx for idx, node_id in enumerate(node_indices)}

        node_features = []
        for node_id in node_indices:
            token_ids = cfg.nodes[node_id]['tokens']
            node_features.append(token_ids)
        # node_features = np.array(node_features)
        node_features = torch.tensor(node_features, dtype=torch.long)

        edge_index = []
        for source, target in cfg.edges():
            source_idx = node_id_map[source]
            target_idx = node_id_map[target]
            edge_index.append([source_idx, target_idx])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        return data

    def tokenize_cfg(self, cfg):
        for node in cfg.nodes():
            tokens = node.tokens
            token_ids = [self.tokenizer.word_index.get(token.lower(), 0) for token in tokens]
            token_ids = pad_sequences([token_ids], maxlen=self.max_len, padding='post')[0]
            node.tokens = token_ids
            cfg.nodes[node]['tokens'] = token_ids
            # print(token_ids)
        return cfg


class GraphClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.hidden_dim = hidden_dim

        # Embedding layer with pre-trained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)

        # Sequence model to process node token sequences
        self.sequence_model = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.num_directions = 2
        # GNN layers
        self.conv1 = GCNConv(hidden_dim*2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x: [num_nodes, seq_len]

        # Process node token sequences
        num_nodes, seq_len = x.size()
        x = self.embedding(x)  # x: [num_nodes, seq_len, embedding_dim]

        # Process sequences with GRU
        h_0 = torch.zeros(self.num_directions, num_nodes, self.hidden_dim, device=x.device)

        # h_n: [num_directions, num_nodes, hidden_dim]
        _, h_n = self.sequence_model(x, h_0)

        # Concatenate the hidden states from both directions

        h_n = h_n.view(self.num_directions, num_nodes, self.hidden_dim)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)  # h_n: [num_nodes, hidden_dim * 2]


        x = h_n  # x: [num_nodes, hidden_dim * num_directions]

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))  # Adjusted input dimension
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)  # x: [batch_size, hidden_dim]

        # Classification
        x = self.fc(x)  # x: [batch_size, num_classes]
        return x



def ast_to_graph(ast):
    cfg_builder = CFGBuilder()
    cfg = cfg_builder.build(ast)
    return cfg

def escape_newlines_in_strings(code):
    # A regex to match strings and escape any newline character within them
    return re.sub(r'(?s)"(.*?)"', lambda m: m.group(0).replace('\n', '\\n'), code)


def strip_comments(c_code):
    # Regular expression to match single-line and multi-line comments
    pattern = r'//.*?$|/\*.*?\*/'
    return re.sub(pattern, '', c_code, flags=re.DOTALL | re.MULTILINE)


def pre_processing(c_code):
    new_code = escape_newlines_in_strings(c_code)
    new_code = strip_comments(new_code)
    new_code = re.sub(r'\\', '', new_code)
    return new_code

def flatten_list(ls):
    output = []
    for item in ls:
        inner_list = [item.value for item in tokenize_code(item)]
        output +=  inner_list
    return output

def load_dataset(folders_amount: int, items_per_folder: int, normalise=False):
    programs_dict = {}
    graph_text_dict = {}
    graph_dict = {}
    parser = c_parser.CParser()
    for i in range(folders_amount):
        folder_path = 'Thesis Data/ProgramData/' + str(i)
        files = glob.glob(os.path.join(folder_path, '*'))
        count = 0
        for file in files:
            with open(file, 'r') as f:
                src_code = pre_processing(f.read())
                if normalise:
                    src_code = get_string(normalise_identifiers(tokenize_code(src_code)))
                try:
                    ast = parser.parse(src_code)
                except:
                    # print("Something went wrong")
                    continue
                g = ast_to_graph(ast)
                node_labels = nx.get_node_attributes(g, 'label')
                if i not in programs_dict.keys():
                    programs_dict[i] = [src_code]
                    raw_list = list(node_labels.values())
                    flattened_list = flatten_list(raw_list)
                    graph_text_dict[i] = [flattened_list]
                    graph_dict[i] = [g]
                else:
                    programs_dict.get(i).append(src_code)
                    raw_list = list(node_labels.values())
                    flattened_list = flatten_list(raw_list)
                    graph_text_dict.get(i).append(flattened_list)
                    graph_dict.get(i).append(g)
            count += 1
            if count >= items_per_folder:
                break

    return programs_dict, graph_text_dict, graph_dict

def display_graph(graph):
    pos = graphviz_layout(graph, prog="circo")
    plt.figure(4, figsize=(10, 6), dpi=240)
    options = {
        "font_size": 8,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 0.5,
        "width": 0.5,
    }
    nx.draw(graph, pos=pos, **options)
    node_labels = nx.get_node_attributes(graph, 'label')
    # print(node_labels)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=6, font_family="sans-serif")
    plt.show()

def iteration(classes: int, items_per_folder: int, normalise: bool):

    programs_dict, gt_dict, graph_dict = load_dataset(classes + 1, items_per_folder, normalise=normalise)

    tokenized_corpus_list = list(gt_dict.values())
    tokenized_corpus = []
    for inner in tokenized_corpus_list: tokenized_corpus += inner

    # Train Word2Vec model
    embedding_dim=100
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=25, min_count=1, workers=4)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_corpus)
    sequences = tokenizer.texts_to_sequences(tokenized_corpus)
    word_index = tokenizer.word_index


    dataset = GraphDataset(graph_dict, tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    word_vectors = word2vec_model.wv
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            # Randomly initialize the embedding for out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))




    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphClassifier(embedding_matrix, hidden_dim=128, num_classes=classes+1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        timer = time.time()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        print(f'Time taken: {time.time() - timer}')



    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            _, predicted = torch.max(out, 1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    accuracy = correct / total
    # print(classification_report(all_labels, all_preds))
    print(f'Validation Accuracy: {accuracy:.4f}')
    return all_preds, all_labels

def program(classes, items_per_folder, epochs, normalise):
    total_predictions = []
    total_labels = []
    for i in range(epochs):
        predictions, labels = iteration(classes, items_per_folder, normalise)
        total_predictions += predictions
        total_labels += labels
    return total_predictions, total_labels


if __name__ == '__main__':
    print('Starting 1')
    no_comments_predictions, no_comments_labels = program(50, 150, 3, False)
    print('Starting 2')
    normalised_predictions, normalised_labels = program(50, 150, 3, True)


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
    scores_nc = [accuracy_nc, precision_nc, recall_nc, f1_nc]
    scores_norm = [accuracy_norm, precision_norm, recall_norm, f1_norm]

    all_scores = [scores_nc, scores_norm]
    conditions = ['No Comments', 'Normalised']  # Names for each plot
    colors = ['blue', 'red']  # blue for nc, red for norm

    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns of plots

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