import networkx as nx


class Tree:
    def __init__(self, graph: nx.DiGraph):
        """
        Initializes a tree with an optional root node.

        Args:
            root (TreeNode): The root node of the tree.
        """
        self.graph = graph
        self.root = 0

    def getchildren(self, node_idx):
        output = []
        for c in self.graph.neighbors(node_idx):
            output.append(c)
        return output

    def has_children(self, node_idx):
        count = 0
        for c in self.graph.neighbors(node_idx):
            count += 1
        return count != 0  # True if has children, False if it has no children

    def get_label(self, node_idx):
        return self.graph.nodes[node_idx]['label']

    def set_label(self, node_idx, new_label):
        self.graph.nodes[node_idx]['label'] = new_label

    def get_nodes_as_list(self):
        def append_node(node):
            output = [self.get_label(node)]
            for c in self.getchildren(node):
                if c == node + 1:
                    output = append_node(c) + output
                else:
                    output += append_node(c)
            return output
        return append_node(0)


