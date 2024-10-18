import time

from pycparser import c_parser
import re
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers.homemade_tokenizerv2 import get_string, normalise_identifiers, tokenize_code
from treefile import Tree
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from gensim.models import Word2Vec
import numpy as np
import glob
import os
import copy


class TreeDataset(Dataset):
    def __init__(self, trees_dict, tokenizer):
        self.trees = []
        self.labels = []
        self.tokenizer = tokenizer
        for label, trees in trees_dict.items():
            self.trees += trees
            self.labels += [label] * len(trees)

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        tree = copy.deepcopy(self.trees[idx])
        label = self.labels[idx]
        tree = self.tokenize_tree(tree)
        return tree, torch.tensor(label, dtype=torch.long)

    def tokenize_tree(self, t):
        def tokenize_node(t: Tree, node):
            t.set_label(node, self.tokenizer.word_index.get(t.get_label(node).lower(), 0))
            for c in t.getchildren(node):
                tokenize_node(t, c)

        tokenize_node(t, 0)
        return t


# ----- TreeLSTM model -----

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
    def __init__(self, input_dim, hidden_dim, device, embedding_matrix, classes):
        super(TreeLSTM, self).__init__()
        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding = self.embedding.to(device)

        self.cell = TreeLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Add a linear layer to classify the hidden state of the root
        self.fc = nn.Linear(hidden_dim, classes + 1).to(device)

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

        # Pass the root hidden state through the classification layer
        logits = self.fc(root_h.to(device))
        return logits



# Preprocess the code: Escape newlines in string literals
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


def ast_to_graph(node, graph=None, parent=None):
    if graph is None:
        graph = nx.DiGraph()

    # Add the current node to the graph
    current_node = len(graph)
    # Extract node type
    node_type = type(node).__name__

    # Customize the label for specific node types to show additional information
    if node_type == 'ID':
        label = f'{node.name}'  # Show variable name for ID nodes
    elif node_type == 'BinaryOp':
        label = f'{node.op}'  # Show operator for BinaryOp nodes
    elif node_type == 'IdentifierType':
        label = f'{node.names[0]}'  # Show the identifier type
    elif node_type == 'TypeDecl':
        label = f'{node.declname}'
    elif node_type == 'Constant':
        label = f'{node.value}'
    elif node_type == 'UnaryOp':
        label = f'{node.op}'
    elif node_type == 'Assignment':
        label = '='
    else:
        label = node_type  # Default label is just the node type

    graph.add_node(current_node, label=label)  # Add node with custom label

    # Add an edge from the parent to the current node
    if parent is not None:
        graph.add_edge(parent, current_node)

    # Recursively traverse child nodes
    if hasattr(node, 'children'):
        for _, child in node.children():
            ast_to_graph(child, graph, current_node)

    return graph

def load_dataset(folders_amount: int, items_per_folder: int, normalise: bool):
    programs_dict = {}
    ast_dict = {}
    tree_graph_dict = {}
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
                    # print('ending up here')
                    continue
                ast_buffer_write = open("../buffer.txt", "w")
                ast.show(ast_buffer_write)
                ast_buffer_write.close()
                ast_buffer_read = open("../buffer.txt", "r")
                astString = ast_buffer_read.read()
                g = ast_to_graph(ast)
                t = Tree(g)
                if i not in programs_dict.keys():
                    programs_dict[i] = [src_code]
                    ast_dict[i] = [astString]
                    tree_graph_dict[i] = [t]
                else:
                    programs_dict.get(i).append(src_code)
                    ast_dict.get(i).append(astString)
                    tree_graph_dict.get(i).append(t)
            count += 1
            if count > + items_per_folder: break

    return programs_dict, ast_dict, tree_graph_dict

def iteration(classes: int, items_per_folder: int, normalise: bool):
    programs_dict, ast_dict, tree_graph_dict = load_dataset(folders_amount=classes + 1, items_per_folder=items_per_folder, normalise=normalise)
    del programs_dict, ast_dict
    # Tokenize the corpus
    tokenized_corpus_list = [[item.get_nodes_as_list() for item in ls] for ls in tree_graph_dict.values()]

    tokenized_corpus = []
    for inner in tokenized_corpus_list: tokenized_corpus += inner

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=25, min_count=1, workers=4)
    # Save the model
    # word2vec_model.save("word2vec.model")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokenized_corpus)
    sequences = tokenizer.texts_to_sequences(tokenized_corpus)
    word_index = tokenizer.word_index

    del tokenized_corpus, tokenized_corpus_list

    dataset = TreeDataset(tree_graph_dict, tokenizer)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    embedding_dim = 100
    word_vectors = word2vec_model.wv
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]
        else:
            # Randomly initialize the embedding for out-of-vocabulary words
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))


    # AST PRINTING
    """graph = ast_to_graph(ast)
    pos = graphviz_layout(graph, prog="dot")
    plt.figure(4, figsize=(16, 6), dpi=240)
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
    # plt.show()
    
    t = Tree(graph)
    
    # Print the AST
    # ast.show()
    """


    # --------------------------Training Starts Here---------------------------------

    # Step 2: Initialize the TreeLSTM model
    input_dim = embedding_dim  # Dimension of the input at each tree node
    hidden_dim = 256  # Hidden dimension size for the TreeLSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeLSTM(input_dim, hidden_dim, device, embedding_matrix, classes).to(device)

    # Step 3: Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Assuming a classification task
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Step 4: Training loop
    epochs = 5
    print('started training...')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        timer = time.time()

        for batch in train_loader:
            # Each batch is a list with a single element (tree, label)
            tree, label = batch[0]
            label = label.to(device)

            # Reset the gradients
            optimizer.zero_grad()

            # Forward pass through TreeLSTM
            output = model(tree, device)  # This should return the root hidden state of the TreeLSTM

            # Compute loss
            loss = criterion(output, label)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        print(f'time taken: {time.time() - timer}')


    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculations for testing
        for batch in test_loader:
            # Each batch is a list with a single element (tree, label)
            tree, label = batch[0]
            label = label.to(device)

            # Forward pass through TreeLSTM
            output = model(tree, device)  # This should return the root hidden state of the TreeLSTM

            # Compute loss
            loss = criterion(output, label)
            test_loss += loss.item()

            # If classification task, calculate accuracy
            pred = output.argmax(dim=-1)  # Get the index of the max log-probability
            correct += (pred == label).sum().item()
            # _, predicted = torch.max(output, 1)
            all_preds.append(pred.item())
            all_labels.append(label.item())

    # Compute average loss and accuracy
    # print(all_preds)
    # print(all_labels)
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # print(classification_report(all_labels, all_preds))
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
    no_comments_predictions, no_comments_labels = program(50, 100, 3, False)
    print('Starting 2')
    normalised_predictions, normalised_labels = program(50, 100, 3, True)


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

    import csv

    # Example lists to write as columns


    # Open a CSV file for writing
    with open('AST_classification_104_300.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Use zip() to combine the lists element-wise and write them as rows
        for row in zip(no_comments_predictions, no_comments_labels, normalised_predictions, normalised_labels):
            writer.writerow(row)

