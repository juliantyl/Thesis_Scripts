import networkx as nx
from pycparser import c_ast, c_generator

from tokenizers.homemade_tokenizerv2 import tokenize_code


class CFGNode:
    def __init__(self, name, visible_label, ast_node=None):
        self.name = name
        self.ast_node = ast_node
        temp = tokenize_code(visible_label)
        self.tokens = [lt.value for lt in temp]

    def __repr__(self):
        return self.name

class CFGBuilder(c_ast.NodeVisitor):
    def __init__(self):
        self.cfg = nx.DiGraph()
        self.current_node = None
        self.node_counter = 0
        self.entry_node = self._new_node("Entry", "Entry")
        self.exit_node = self._new_node("Exit", "Exit")
        self.last_node = self.entry_node

    def _new_node(self, label, visible_label="", ast_node=None):
        node = CFGNode(f"{label}_{self.node_counter}", visible_label, ast_node)
        self.node_counter += 1
        self.cfg.add_node(node, label=visible_label)
        return node

    def _add_edge(self, from_node, to_node):
        self.cfg.add_edge(from_node, to_node)

    def build(self, ast):
        self.visit(ast)
        self._add_edge(self.last_node, self.exit_node)
        return self.cfg

    def visit_FuncDef(self, node):
        self.current_node = self.entry_node
        self.visit(node.body)
        self._add_edge(self.last_node, self.exit_node)

    def visit_Compound(self, node):
        for stmt in node.block_items or []:
            self.visit(stmt)

    def visit_If(self, node):
        generator = c_generator.CGenerator()
        cond_code = f'If {generator.visit(node.cond)}'
        cond_node = self._new_node(f"IfCondition", cond_code, node.cond)
        self._add_edge(self.last_node, cond_node)

        # True branch
        self.last_node = cond_node
        self.visit(node.iftrue)
        true_exit = self.last_node

        # False branch
        if node.iffalse:
            self.last_node = cond_node
            self.visit(node.iffalse)
            false_exit = self.last_node
        else:
            false_exit = cond_node

        # Merge point
        label = "IfMerge"
        merge_node = self._new_node("IfMerge", label)
        self._add_edge(true_exit, merge_node)
        self._add_edge(false_exit, merge_node)
        self.last_node = merge_node

    def visit_Assignment(self, node):
        generator = c_generator.CGenerator()
        label2 = generator.visit(node)
        assign_node = self._new_node(f"Assign_{node.lvalue}", label2, node)
        self._add_edge(self.last_node, assign_node)
        self.last_node = assign_node

    def visit_Decl(self, node):
        generator = c_generator.CGenerator()
        label = generator.visit(node)
        decl_node = self._new_node(f"Decl_{node.name}", label, node)
        self._add_edge(self.last_node, decl_node)
        self.last_node = decl_node
        if node.init:
            self.visit_Assignment(c_ast.Assignment(op='=', lvalue=c_ast.ID(name=node.name), rvalue=node.init))

    def visit_Return(self, node):
        generator = c_generator.CGenerator()
        label = generator.visit(node)
        return_node = self._new_node("Return", label, node)
        self._add_edge(self.last_node, return_node)
        self._add_edge(return_node, self.exit_node)
        self.last_node = return_node

    def visit_While(self, node):
        generator = c_generator.CGenerator()
        label = generator.visit(node.cond)
        cond_node = self._new_node("WhileCondition", label, node.cond)
        self._add_edge(self.last_node, cond_node)
        self.last_node = cond_node

        # Loop body
        label = "WhileBodyStart"
        body_start_node = self._new_node("WhileBodyStart", label)
        self._add_edge(cond_node, body_start_node)
        self.last_node = body_start_node
        self.visit(node.stmt)
        self._add_edge(self.last_node, cond_node)  # Loop back to condition

        # Exit loop
        label = "WhiteExit"
        loop_exit_node = self._new_node("WhileExit", label)
        self._add_edge(cond_node, loop_exit_node)
        self.last_node = loop_exit_node

    def visit_For(self, node):
        generator = c_generator.CGenerator()
        label = generator.visit(node.init)
        init_node = self._new_node("ForInit", label, node.init)
        self._add_edge(self.last_node, init_node)
        self.last_node = init_node

        label = generator.visit(node.cond)
        cond_node = self._new_node("ForCondition", label, node.cond)
        self._add_edge(self.last_node, cond_node)

        # Loop body
        label = "ForBodyStart"
        body_start_node = self._new_node("ForBodyStart", label)
        self._add_edge(cond_node, body_start_node)
        self.last_node = body_start_node
        self.visit(node.stmt)

        # Loop increment
        label = "next"
        next_node = self._new_node("ForNext", label, node.next)
        self._add_edge(self.last_node, next_node)
        self._add_edge(next_node, cond_node)  # Loop back to condition

        # Exit loop
        label = "ForExit"
        loop_exit_node = self._new_node("ForExit", label)
        self._add_edge(cond_node, loop_exit_node)
        self.last_node = loop_exit_node

    def visit_FuncCall(self, node):
        if hasattr(node.name, 'name'):
            label = f'Call {node.name.name}'
        else:
            label = f'Call {node.name}'
        func_call_node = self._new_node(label, label, node)
        self._add_edge(self.last_node, func_call_node)
        self.last_node = func_call_node

    def generic_visit(self, node):
        for c_name, c in node.children():
            self.visit(c)

    def get_nodes(self):
        return list(self.cfg.nodes())

