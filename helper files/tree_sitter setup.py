import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

JAVA_LANGUAGE = Language(tsjava.language())

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
    node_id = id(node)
    # Add the node to the graph with attributes
    node_type = node.type.replace('"', r'\"')
    node_type = node.type.replace('.', r'dot')
    label = f"{node_type}"
    if node.is_named:
        if label == 'identifier':
            label = get_node_text(node, src_code)
        elif label == 'type_identifier':
            label = get_node_text(node, src_code)
    graph.add_node(node_id, label=label)
    # If there is a parent node, add an edge from the parent to the current node
    if parent_id is not None:
        graph.add_edge(parent_id, node_id)
    # Recursively process the children
    for child in node.children:
        build_graph(child, graph, src_code, node_id)


code = '''
public void run () {
    new UIJob (Messages.AMLFinishWorkerVariant_3) {
        @Override
        public IStatus runInUIThread (final IProgressMonitor myMon) {
            try {
                mapView.saveMap (myMon, m_data.getMapFile ());
            } catch (final CoreException e) {
                NofdpCorePlugin.getDefault ().getLog ().log (StatusUtilities.statusFromThrowable (e));
                return new Status (IStatus.ERROR, ""AMLFinishWorkerVariant"", Messages.AMLFinishWorkerVariant_5 + e.getMessage ());
            }
            return Status.OK_STATUS;
        }}

    .schedule ();
}
'''

tree = parse_java_code(code)
# print_tree(tree.root_node)
graph = nx.DiGraph()
build_graph(tree.root_node, graph, code)

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

