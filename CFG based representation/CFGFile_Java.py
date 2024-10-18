import networkx as nx
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node
from networkx.drawing.nx_pydot import graphviz_layout


JAVA_LANGUAGE = Language(tsjava.language())

def get_node_text(node, src_code):
    return src_code[node.start_byte:node.end_byte]


class CFGBuilder:
    def __init__(self, src_code):
        self.src_code = src_code
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.parser = Parser(JAVA_LANGUAGE)
        self.tree = self.parser.parse(bytes(src_code, 'utf8'))
        self.entry_node = self._new_node("Entry")
        self.exit_node = self._new_node("Exit")
        self.current_node = self.entry_node  # Start from the entry node

    def _new_node(self, label, ast_node=None):
        node_id = self.node_counter
        self.node_counter += 1
        # Replace colons in label
        label = label.replace(':', ';')
        self.graph.add_node(node_id, label=label, ast_node=ast_node)
        return node_id

    def _add_edge(self, from_node, to_node):
        self.graph.add_edge(from_node, to_node)

    def build(self):
        # Start building the CFG from the root node
        self.build_cfg(self.tree.root_node, self.current_node)
        # Connect the last node to the exit node if not already connected
        if not self.graph.has_edge(self.current_node, self.exit_node):
            self._add_edge(self.current_node, self.exit_node)
        return self.graph

    def build_cfg(self, node, parent_id):
        if node is None:
            print("Warning: node is None")
            return parent_id
        print(f"Visiting node type: {node.type}")
        method_name = f"handle_{node.type}"
        handler = getattr(self, method_name, self.generic_handler)
        return handler(node, parent_id)



    def handle_method_declaration(self, node, parent_id):
        # We start a new CFG from method declarations
        body_node = node.child_by_field_name('body')
        if body_node:
            return self.build_cfg(body_node, parent_id)
        else:
            return parent_id

    def handle_block(self, node, parent_id):
        last_node = parent_id
        for child in node.children:
            if child.is_named:
                last_node = self.build_cfg(child, last_node)
        return last_node

    def handle_expression_statement(self, node, parent_id):
        expr_text = get_node_text(node, self.src_code).strip()
        node_id = self._new_node(expr_text, node)
        self._add_edge(parent_id, node_id)
        last_node = node_id
        for child in node.children:
            if child.is_named:
                last_node = self.build_cfg(child, last_node)
        return last_node

    def handle_return_statement(self, node, parent_id):
        return_text = get_node_text(node, self.src_code).strip()
        node_id = self._new_node(return_text, node)
        self._add_edge(parent_id, node_id)
        # Return statements terminate the control flow
        self._add_edge(node_id, self.exit_node)
        return node_id  # This path ends here

    def handle_if_statement(self, node, parent_id):
        # Condition
        condition_node = node.child_by_field_name('condition')
        condition_text = get_node_text(condition_node, self.src_code).strip()
        condition_id = self._new_node(f"If {condition_text}", condition_node)
        self._add_edge(parent_id, condition_id)

        # Consequence
        consequence_node = node.child_by_field_name('consequence')
        true_exit = self.build_cfg(consequence_node, condition_id)

        # Alternative
        alternative_node = node.child_by_field_name('alternative')
        if alternative_node:
            false_exit = self.build_cfg(alternative_node, condition_id)
        else:
            false_exit = condition_id  # If no else, the false path continues from condition

        # Merge point
        merge_id = self._new_node("Merge", node)
        self._add_edge(true_exit, merge_id)
        if false_exit != condition_id:
            self._add_edge(false_exit, merge_id)
        else:
            self._add_edge(condition_id, merge_id)

        return merge_id

    def handle_try_statement(self, node, parent_id):
        try_node = self._new_node("Try", node)
        self._add_edge(parent_id, try_node)

        # Try block
        try_block = node.child_by_field_name('body')
        try_exit = self.build_cfg(try_block, try_node)

        # Collect exits from catch clauses
        catch_exits = []
        for child in node.children:
            if child.type == 'catch_clause':
                catch_exit = self.build_cfg(child, try_node)
                catch_exits.append(catch_exit)

        # Finally block (if present)
        finally_node = node.child_by_field_name('finally')
        if finally_node:
            finally_exit = self.build_cfg(finally_node, try_node)
            merge_id = finally_exit
        else:
            # Merge exits from try and catch blocks
            merge_id = self._new_node("Merge", node)
            self._add_edge(try_exit, merge_id)
            for exit_node in catch_exits:
                self._add_edge(exit_node, merge_id)

        return merge_id

    def handle_catch_clause(self, node, parent_id):
        catch_param = node.child_by_field_name('parameter')
        catch_text = f"Catch {get_node_text(catch_param, self.src_code).strip()}"
        catch_id = self._new_node(catch_text, node)
        self._add_edge(parent_id, catch_id)

        # Catch body
        body_node = node.child_by_field_name('body')
        exit_id = self.build_cfg(body_node, catch_id)
        return exit_id

    def handle_method_invocation(self, node, parent_id):
        method_name_node = node.child_by_field_name('name')
        method_name = get_node_text(method_name_node, self.src_code).strip()
        method_node_id = self._new_node(f"Call {method_name}", node)
        self._add_edge(parent_id, method_node_id)
        last_node = method_node_id
        # Process arguments
        arguments_node = node.child_by_field_name('arguments')
        if arguments_node:
            for arg in arguments_node.named_children:
                last_node = self.build_cfg(arg, last_node)
        return last_node

    def handle_object_creation_expression(self, node, parent_id):
        type_node = node.child_by_field_name('type')
        type_name = get_node_text(type_node, self.src_code).strip() if type_node else "AnonymousClass"
        creation_node_id = self._new_node(f"New {type_name}", node)
        self._add_edge(parent_id, creation_node_id)
        last_node = creation_node_id
        # Process arguments
        arguments_node = node.child_by_field_name('arguments')
        if arguments_node:
            for arg in arguments_node.named_children:
                last_node = self.build_cfg(arg, last_node)
        # Process class body (anonymous class)
        class_body_node = node.child_by_field_name('body')
        if class_body_node:
            last_node = self.build_cfg(class_body_node, last_node)
        return last_node

    def handle_class_body(self, node, parent_id):
        last_node = parent_id
        for child in node.children:
            if child.is_named:
                last_node = self.build_cfg(child, last_node)
        return last_node


    def generic_handler(self, node, parent_id):
        last_node = parent_id
        for child in node.children:
            if child.is_named:
                # Process the child node
                child_label = get_node_text(child, self.src_code).strip()
                child_node_id = self._new_node(child_label, child)
                self._add_edge(last_node, child_node_id)
                # Recursively process the child node
                last_node = self.build_cfg(child, child_node_id)
        return last_node

