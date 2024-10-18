import random
import time

import tree_sitter_java as tsjava
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tree_sitter import Language, Parser
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from treefile import Tree
import copy

JAVA_LANGUAGE = Language(tsjava.language())


class TreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(input_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_children, c_children, device):
        # Tree-LSTM logic
        if h_children is None or c_children is None:
            h_children = torch.zeros(1, self.hidden_dim).to(device)
            c_children = torch.zeros(1, self.hidden_dim).to(device)
        h_sum = torch.sum(h_children, dim=0).to(device)
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        W_f_x = self.W_f(x).unsqueeze(0).expand(h_children.size(0), -1)  # Expand to match h_children's batch size
        f = torch.sigmoid(W_f_x + self.U_f(h_children))
        c = i * u + torch.sum(f * c_children, dim=0).to(device)
        h = o * torch.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_matrix, device):
        super(TreeLSTM, self).__init__()
        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding = self.embedding.to(device)

        self.cell = TreeLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, tree, device):
        # Recursively apply TreeLSTM to all nodes in the tree
        def recurse(node):
            if not tree.has_children(node):  # Leaf node
                label_index = tree.get_label(node)
                label_index = torch.tensor([label_index], dtype=torch.long).to(device)
                label_embedding = self.embedding(label_index).squeeze(0)
                return self.cell(label_embedding, None, None, device)
            h_children, c_children = zip(*[recurse(child) for child in tree.getchildren(node)])
            h_children = torch.stack(h_children).to(device)
            c_children = torch.stack(c_children).to(device)

            label_index = tree.get_label(node)
            label_index = torch.tensor([label_index], dtype=torch.long).to(device)
            label_embedding = self.embedding(label_index).squeeze(0)
            return self.cell(label_embedding, h_children, c_children, device)

        root_node = tree.root
        root_h, _ = recurse(root_node)
        return root_h


class SiameseTreeNetwork(nn.Module):
    def __init__(self, embedding_matrix, input_dim, hidden_dim, device):
        super(SiameseTreeNetwork, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.device = device

        # Initialize the TreeLSTM model
        self.tree_lstm = TreeLSTM(input_dim, hidden_dim, embedding_matrix, device)

        # Define the fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward_once(self, x):
        # Encode one tree input using TreeLSTM
        root_h = self.tree_lstm(x, self.device)
        return root_h  # root_h is the hidden state at the root

    def forward(self, input1, input2):
        # Forward both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Compute distance (e.g., cosine similarity or difference)
        distance = torch.abs(output1 - output2)
        # Pass through a fully connected layer for binary classification
        out = self.fc(distance)
        return torch.sigmoid(out)


class TreeDataset(Dataset):
    def __init__(self, trees_dict, clones_list, false_pos_list, tokenizer):
        self.inputs = []
        self.labels = []
        self.trees_dict = trees_dict
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
        func_id1 = self.inputs[idx][0]
        func_id2 = self.inputs[idx][1]
        tree1 = copy.deepcopy(self.trees_dict[func_id1])
        tree2 = copy.deepcopy(self.trees_dict[func_id2])
        label = self.labels[idx]
        tree1 = self.tokenize_tree(tree1)
        tree2 = self.tokenize_tree(tree2)
        return tree1, tree2, torch.tensor(label, dtype=torch.long)

    def tokenize_tree(self, t):
        def tokenize_node(t: Tree, node):
            t.set_label(node, self.tokenizer.word_index.get(t.get_label(node).lower(), 0))
            for c in t.getchildren(node):
                tokenize_node(t, c)

        tokenize_node(t, 0)
        return t

def load_data(samples, clone_percentage, normalise):
    if normalise:
        pretty_printed = pd.read_csv('../Thesis Data/BCB/pretty_printed_consistent_renamed_functions.csv', encoding='utf-8',
                                     index_col='function_id')
    else:
        pretty_printed = pd.read_csv('../Thesis Data/BCB/pretty_printed_functions.csv', encoding='utf-8',
                                     index_col='function_id')
    clones = pd.read_csv('../Thesis Data/BCB/clones.csv', encoding='utf-8')
    false_positives = pd.read_csv('../Thesis Data/BCB/false_positives.csv', encoding='utf-8')

    functions = set()

    # Samples from each 'clones' and 'false_positives'
    clones_count = int(clone_percentage * samples)
    fp_count = samples - clones_count

    # Prepare the clones_list
    var = list(zip(clones['function_id_one'], clones['function_id_two']))
    clones_list = random.choices(var, k=clones_count)

    # Prepare the false_pos_list
    var = list(zip(false_positives['function_id_one'], false_positives['function_id_two']))
    false_pos_list = random.choices(var, k=fp_count)

    # Filter clones_list
    valid_clones_list = []
    for id1, id2 in clones_list:
        if id1 in pretty_printed.index and id2 in pretty_printed.index:
            valid_clones_list.append((id1, id2))
            functions.update([id1, id2])  # Add both IDs to the functions set
        else:
            print(f'{id1} or {id2} not found in pretty_printed. Removing from clones_list.')

    # Replace clones_list with the filtered list
    clones_list = valid_clones_list

    # Filter false_pos_list
    valid_false_pos_list = []
    for id1, id2 in false_pos_list:
        if id1 in pretty_printed.index and id2 in pretty_printed.index:
            valid_false_pos_list.append((id1, id2))
            functions.update([id1, id2])  # Add both IDs to the functions set
        else:
            print(f'{id1} or {id2} not found in pretty_printed. Removing from false_pos_list.')

    # Replace false_pos_list with the filtered list
    false_pos_list = valid_false_pos_list

    # Return the results
    return pretty_printed, clones_list, false_pos_list, functions

def get_node_text(node, code):
    return code[node.start_byte:node.end_byte]

def parse_java_code(code):
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    return tree

def print_tree(node, indent=0):
    print('  ' * indent + f"{node.type} [{node.start_point}-{node.end_point}]")
    for child in node.children:
        print_tree(child, indent + 1)

def build_graph(node, graph, src_code, parent_id=None):
    # Create a unique identifier for the current node
    current_node = len(graph)
    # Add the node to the graph with attributes
    node_type = node.type.replace('.', r'dot')
    label = f"{node_type}"
    if node.is_named:
        if label == 'identifier':
            label = get_node_text(node, src_code)
        elif label == 'type_identifier':
            label = get_node_text(node, src_code)
    graph.add_node(current_node, label=label)
    # If there is a parent node, add an edge from the parent to the current node
    if parent_id is not None:
        graph.add_edge(parent_id, current_node)
    # Recursively process the children
    for child in node.children:
        build_graph(child, graph, src_code, current_node)

def create_tree(src_code: str) -> Tree:
    tree = parse_java_code(src_code)
    graph = nx.DiGraph()
    build_graph(tree.root_node, graph, src_code)
    # visualise_graph(graph)
    output_tree = Tree(graph)
    return output_tree

def visualise_graph(graph):
    # Visualize the AST as a NetworkX graph
    pos = nx.spring_layout(graph)  # Positioning algorithm
    labels = nx.get_node_attributes(graph, 'label')


    pos = graphviz_layout(graph, prog="dot")
    plt.figure(1, figsize=(24, 9), dpi=240)
    options = {
        "font_size": 8,
        "node_size": 200,
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



def iteration(samples, clone_percentage, normalise):
    # Reading in files from appropriate csvs and storing them in memory to access later
    pretty_printed, clones, false_positives, functions = load_data(samples, clone_percentage, normalise)
    """print(len(clones))
    print(len(false_positives))"""

    tokenized_corpus = []
    tree_dict = {}
    for function in functions:
        print(function)
        tree = create_tree(pretty_printed.at[function, 'text'])
        tokenized_corpus.append(tree.get_nodes_as_list())
        tree_dict[function] = tree

    # embedding_dim is the size of the vector
    embedding_dim = 150

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_dim, window=25, min_count=1, workers=4,
                              epochs=10)
    # Save the model
    word2vec_model.save("word2vec_java.model")

    # setting up the tokenizer to store each word as an index
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_corpus)
    sequences = tokenizer.texts_to_sequences(tokenized_corpus)
    word_index = tokenizer.word_index

    dataset = TreeDataset(tree_dict, clones, false_positives, tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    word_vectors = word2vec_model.wv
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            # Randomly initialize the embedding for out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = embedding_dim
    hidden_dim = 256
    model = SiameseTreeNetwork(embedding_matrix, input_dim, hidden_dim, device).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    print("Training Start!")
    for epoch in range(num_epochs):
        timer = time.time()
        model.train()
        total_loss = 0
        for batch in train_loader:

            tree1, tree2, label = batch[0]
            label = label.to(device).float()

            # Ensure label is the same shape as output
            if label.dim() == 0:
                label = label.unsqueeze(0)

            optimizer.zero_grad()
            output = model(tree1, tree2)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        print(f'Time taken: {time.time() - timer}')


    # Testing
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            tree1, tree2, label = batch[0]
            label = label.to(device)


            output = model(tree1, tree2)
            predicted = (output >= 0.5).float()
            all_preds.append(int(predicted.item()))
            all_labels.append(label.item())
            total += 1
            correct += (predicted == label).sum().item()

    print(all_preds)
    print(all_labels)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    # print(classification_report(all_labels, all_preds))
    return all_preds, all_labels

def program(epochs, samples, clone_percentage, normalise):
    total_predictions = []
    total_labels = []
    for i in range(epochs):
        predictions, labels = iteration(samples, clone_percentage, normalise)
        total_predictions += predictions
        total_labels += labels
    return total_predictions, total_labels

if __name__ == '__main__':
    print('Starting 1')
    no_comments_predictions50, no_comments_labels50 = program(2, 500, 0.5, False)
    print('Starting 2')
    normalised_predictions50, normalised_labels50 = program(2, 500, 0.5, True)
    print('Starting 3')
    no_comments_predictions25, no_comments_labels25 = program(2, 500, 0.25, False)
    print('Starting 4')
    normalised_predictions25, normalised_labels25 = program(2, 500, 0.25, True)
    print('Starting 5')
    no_comments_predictions75, no_comments_labels75 = program(2, 500, 0.75, False)
    print('Starting 6')
    normalised_predictions75, normalised_labels75 = program(2, 500, 0.75, True)

    accuracy_nc = accuracy_score(no_comments_predictions50, no_comments_labels50)
    precision_nc = precision_score(no_comments_predictions50, no_comments_labels50, average='macro')
    recall_nc = recall_score(no_comments_predictions50, no_comments_labels50, average='macro')
    f1_nc = f1_score(no_comments_predictions50, no_comments_labels50, average='macro')

    accuracy_norm = accuracy_score(normalised_predictions50, normalised_labels50)
    precision_norm = precision_score(normalised_predictions50, normalised_labels50, average='macro')
    recall_norm = recall_score(normalised_predictions50, normalised_labels50, average='macro')
    f1_norm = f1_score(normalised_predictions50, normalised_labels50, average='macro')

    accuracy_nc25 = accuracy_score(no_comments_predictions25, no_comments_labels25)
    precision_nc25 = precision_score(no_comments_predictions25, no_comments_labels25, average='macro')
    recall_nc25 = recall_score(no_comments_predictions25, no_comments_labels25, average='macro')
    f1_nc25 = f1_score(no_comments_predictions25, no_comments_labels25, average='macro')

    accuracy_norm25 = accuracy_score(normalised_predictions25, normalised_labels25)
    precision_norm25 = precision_score(normalised_predictions25, normalised_labels25, average='macro')
    recall_norm25 = recall_score(normalised_predictions25, normalised_labels25, average='macro')
    f1_norm25 = f1_score(normalised_predictions25, normalised_labels25, average='macro')

    accuracy_nc75 = accuracy_score(no_comments_predictions75, no_comments_labels75)
    precision_nc75 = precision_score(no_comments_predictions75, no_comments_labels75, average='macro')
    recall_nc75 = recall_score(no_comments_predictions75, no_comments_labels75, average='macro')
    f1_nc75 = f1_score(no_comments_predictions75, no_comments_labels75, average='macro')

    accuracy_norm75 = accuracy_score(normalised_predictions75, normalised_labels75)
    precision_norm75 = precision_score(normalised_predictions75, normalised_labels75, average='macro')
    recall_norm75 = recall_score(normalised_predictions75, normalised_labels75, average='macro')
    f1_norm75 = f1_score(normalised_predictions75, normalised_labels75, average='macro')

    # Metrics and their corresponding names
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores_nc = [accuracy_nc, precision_nc, recall_nc, f1_nc]
    scores_norm = [accuracy_norm, precision_norm, recall_norm, f1_norm]

    scores_nc25 = [accuracy_nc25, precision_nc25, recall_nc25, f1_nc25]
    scores_norm25 = [accuracy_norm25, precision_norm25, recall_norm25, f1_norm25]

    scores_nc75 = [accuracy_nc75, precision_nc75, recall_nc75, f1_nc75]
    scores_norm75 = [accuracy_norm75, precision_norm75, recall_norm75, f1_norm75]

    all_scores = [scores_nc, scores_norm]
    all_scores25 = [scores_nc25, scores_norm25]
    all_scores75 = [scores_nc75, scores_norm75]
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

    # Create subplots: 1 row, 3 columns
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns of plots

    # Loop through each condition and corresponding scores to plot them
    for i, ax in enumerate(axes3):
        ax.bar(metrics, all_scores25[i], color=colors[i])
        print(all_scores25[i])
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(conditions[i], fontsize=16)
        ax.set_ylabel('score', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Create subplots: 1 row, 3 columns
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns of plots

    # Loop through each condition and corresponding scores to plot them
    for i, ax in enumerate(axes4):
        ax.bar(metrics, all_scores75[i], color=colors[i])
        print(all_scores75[i])
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(conditions[i], fontsize=16)
        ax.set_ylabel('SCore', fontsize=12)
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

    fig5, axes5 = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns for 4 metrics

    # Loop through each metric and plot raw, nc, and norm as bars with the specified colors
    for i, ax in enumerate(axes5):
        ax.bar(conditions, [scores_nc25[i], scores_norm25[i]], color=colors)
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(metrics[i], fontsize=16)
        ax.set_ylabel('score', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout for the second set of plots
    plt.tight_layout()

    fig6, axes6 = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns for 4 metrics

    # Loop through each metric and plot raw, nc, and norm as bars with the specified colors
    for i, ax in enumerate(axes6):
        ax.bar(conditions, [scores_nc75[i], scores_norm75[i]], color=colors)
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(metrics[i], fontsize=16)
        ax.set_ylabel('SCore', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout for the second set of plots
    plt.tight_layout()

    # Show both sets of plots
    plt.show()

    # Confustion Matrix
    cm = confusion_matrix(normalised_predictions50, normalised_labels50)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Normalised Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(no_comments_predictions50, no_comments_labels50)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('No comments')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(normalised_predictions25, normalised_labels25)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Normalised Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(no_comments_predictions25, no_comments_labels25)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('No comments')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(normalised_predictions75, normalised_labels75)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Normalised Code')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    cm = confusion_matrix(no_comments_predictions75, no_comments_labels75)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('No comments')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
