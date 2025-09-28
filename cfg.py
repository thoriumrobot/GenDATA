#!/usr/bin/env python3
"""
CFG Generation Script with Best Practices Defaults

This script generates Control Flow Graphs (CFGs) with:
- Dataflow information by default
- Enhanced edge types for control vs dataflow
- Consistent integration with training and prediction pipelines

Best Practices:
- Always includes dataflow edges connecting variables of the same name
- Uses different edge types for control flow vs dataflow
- Maintains consistency with model training requirements
- Provides comprehensive graph information for better predictions
"""

import networkx as nx
import javalang
import json
import os
from javalang.parse import parse
from javalang.tree import *

def parse_java_file(java_file_path):
    """
    Parses a Java file and returns the Abstract Syntax Tree (AST).
    """
    try:
        with open(java_file_path, 'r') as file:
            java_code = file.read()
        tree = parse(java_code)
        return tree
    except Exception as e:
        print(f"Warning: Could not parse {java_file_path}: {e}")
        return None

def extract_method_declarations(parsed_java_code):
    """
    Extracts method declarations from the parsed Java code.
    """
    if parsed_java_code is None:
        return []
    method_declarations = []
    for _, node in parsed_java_code.filter(MethodDeclaration):
        method_declarations.append(node)
    return method_declarations

def extract_variables_from_expression(expression):
    """
    Extracts variable names from an expression.
    Returns a tuple: (defined_vars, used_vars)
    """
    defined_vars = set()
    used_vars = set()
    
    if expression is None:
        return defined_vars, used_vars
    
    # Handle assignment expressions
    if isinstance(expression, Assignment):
        # Left side defines variables
        if isinstance(expression.expressionl, MemberReference):
            defined_vars.add(expression.expressionl.member)
        
        # Right side uses variables
        def_vars, use_vars = extract_variables_from_expression(expression.value)
        defined_vars.update(def_vars)
        used_vars.update(use_vars)
    
    # Handle member references (variable uses)
    elif isinstance(expression, MemberReference):
        used_vars.add(expression.member)
    
    # Handle method invocations
    elif isinstance(expression, MethodInvocation):
        # Method name is used
        if expression.member:
            used_vars.add(expression.member)
        # Arguments may use variables
        if expression.arguments:
            for arg in expression.arguments:
                def_vars, use_vars = extract_variables_from_expression(arg)
                defined_vars.update(def_vars)
                used_vars.update(use_vars)
    
    # Handle binary operations
    elif isinstance(expression, BinaryOperation):
        def_vars1, use_vars1 = extract_variables_from_expression(expression.operandl)
        def_vars2, use_vars2 = extract_variables_from_expression(expression.operandr)
        defined_vars.update(def_vars1)
        defined_vars.update(def_vars2)
        used_vars.update(use_vars1)
        used_vars.update(use_vars2)
    
    # Handle array access
    elif isinstance(expression, ArraySelector):
        def_vars, use_vars = extract_variables_from_expression(expression.array)
        defined_vars.update(def_vars)
        used_vars.update(use_vars)
        def_vars2, use_vars2 = extract_variables_from_expression(expression.index)
        defined_vars.update(def_vars2)
        used_vars.update(use_vars2)
    
    # Handle ternary operations
    elif isinstance(expression, TernaryExpression):
        def_vars1, use_vars1 = extract_variables_from_expression(expression.condition)
        def_vars2, use_vars2 = extract_variables_from_expression(expression.if_true)
        def_vars3, use_vars3 = extract_variables_from_expression(expression.if_false)
        defined_vars.update(def_vars1)
        defined_vars.update(def_vars2)
        defined_vars.update(def_vars3)
        used_vars.update(use_vars1)
        used_vars.update(use_vars2)
        used_vars.update(use_vars3)
    
    return defined_vars, used_vars

def create_cfg(method, source_lines):
    """
    Creates a Control Flow Graph (CFG) with dataflow information for a given method.
    """
    cfg = nx.MultiDiGraph()  # Use MultiDiGraph to support different edge types
    entry_node = 'Entry'
    exit_node = 'Exit'
    cfg.add_node(entry_node, label='Entry', node_type='control')
    cfg.add_node(exit_node, label='Exit', node_type='control')

    # Track variable definitions and uses for dataflow analysis
    variable_definitions = {}  # var_name -> (node_id, line_number)
    variable_uses = []  # List of (node_id, var_name, line_number)

    if method.body:
        # Start processing the method body from the entry node
        last_nodes = process_block_statements(method.body, cfg, entry_node, source_lines, variable_definitions, variable_uses)
    else:
        last_nodes = [entry_node]

    # Connect the last nodes to the exit node
    for node in last_nodes:
        cfg.add_edge(node, exit_node, edge_type='control')

    # Add dataflow edges
    add_dataflow_edges(cfg, variable_definitions, variable_uses)

    return cfg

def add_dataflow_edges(cfg, variable_definitions, variable_uses):
    """
    Adds dataflow edges connecting variable definitions to their uses.
    """
    for use_node_id, var_name, use_line in variable_uses:
        if var_name in variable_definitions:
            def_node_id, def_line = variable_definitions[var_name]
            # Only add dataflow edge if definition comes before use
            if def_line is not None and use_line is not None and def_line < use_line:
                cfg.add_edge(def_node_id, use_node_id, edge_type='dataflow', variable=var_name)

def process_block_statements(block, cfg, current_node, source_lines, variable_definitions, variable_uses):
    """
    Processes a block of statements and adds them to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    last_nodes = [current_node]
    # javalang can represent a block as a BlockStatement with .statements or directly as a list
    stmts = None
    if hasattr(block, 'statements'):
        stmts = block.statements
    elif isinstance(block, list):
        stmts = block
    else:
        stmts = []
    for statement in stmts:
        new_last_nodes = []
        for node in last_nodes:
            nodes = process_statement(statement, cfg, node, source_lines, variable_definitions, variable_uses)
            new_last_nodes.extend(nodes)
        last_nodes = new_last_nodes
    return last_nodes

def process_statement(statement, cfg, current_node, source_lines, variable_definitions, variable_uses):
    """
    Processes a single statement and adds it to the CFG starting from current_node.
    Returns a list of last nodes.
    """
    if isinstance(statement, BlockStatement):
        # Process nested block statements
        return process_block_statements(statement, cfg, current_node, source_lines, variable_definitions, variable_uses)

    elif isinstance(statement, IfStatement):
        # Handle if statement
        condition_node = f'If({statement.condition})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(condition_node, label=f'If({statement.condition})', line=line_num, node_type='control')
        cfg.add_edge(current_node, condition_node, edge_type='control')
        
        # Extract variables from condition
        def_vars, use_vars = extract_variables_from_expression(statement.condition)
        for var in def_vars:
            variable_definitions[var] = (condition_node, line_num)
        for var in use_vars:
            variable_uses.append((condition_node, var, line_num))

        # Process the 'then' block
        then_last_nodes = process_statement(statement.then_statement, cfg, condition_node, source_lines, variable_definitions, variable_uses)

        # Process the 'else' block if present
        if statement.else_statement:
            else_last_nodes = process_statement(statement.else_statement, cfg, condition_node, source_lines, variable_definitions, variable_uses)
        else:
            else_last_nodes = [condition_node]

        # Merge the paths
        return then_last_nodes + else_last_nodes

    elif isinstance(statement, WhileStatement):
        # Handle while loop
        condition_node = f'While({statement.condition})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(condition_node, label=f'While({statement.condition})', line=line_num, node_type='control')
        cfg.add_edge(current_node, condition_node, edge_type='control')
        
        # Extract variables from condition
        def_vars, use_vars = extract_variables_from_expression(statement.condition)
        for var in def_vars:
            variable_definitions[var] = (condition_node, line_num)
        for var in use_vars:
            variable_uses.append((condition_node, var, line_num))

        # Process the body of the while loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node, source_lines, variable_definitions, variable_uses)

        # Add edge from body last nodes back to condition node
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node, edge_type='control')

        # The loop can exit after checking the condition
        return [condition_node]

    elif isinstance(statement, ForStatement):
        # Handle for loop
        condition_str = str(statement.control.condition) if statement.control and statement.control.condition else ""
        condition_node = f'For({condition_str})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(condition_node, label=f'For({condition_str})', line=line_num, node_type='control')
        cfg.add_edge(current_node, condition_node, edge_type='control')
        
        # Extract variables from condition
        if statement.control and statement.control.condition:
            def_vars, use_vars = extract_variables_from_expression(statement.control.condition)
            for var in def_vars:
                variable_definitions[var] = (condition_node, line_num)
            for var in use_vars:
                variable_uses.append((condition_node, var, line_num))

        # Process the body of the for loop
        body_last_nodes = process_statement(statement.body, cfg, condition_node, source_lines, variable_definitions, variable_uses)

        # Handle loop updates
        if statement.control and statement.control.update:
            update_str = ", ".join(map(str, statement.control.update)) if isinstance(statement.control.update, list) else str(statement.control.update)
            update_node = f'Update({update_str})'
            cfg.add_node(update_node, label=update_node, line=line_num, node_type='control')
            for node in body_last_nodes:
                cfg.add_edge(node, update_node, edge_type='control')
            cfg.add_edge(update_node, condition_node, edge_type='control')
            
            # Extract variables from update expressions
            if isinstance(statement.control.update, list):
                for update_expr in statement.control.update:
                    def_vars, use_vars = extract_variables_from_expression(update_expr)
                    for var in def_vars:
                        variable_definitions[var] = (update_node, line_num)
                    for var in use_vars:
                        variable_uses.append((update_node, var, line_num))
            else:
                def_vars, use_vars = extract_variables_from_expression(statement.control.update)
                for var in def_vars:
                    variable_definitions[var] = (update_node, line_num)
                for var in use_vars:
                    variable_uses.append((update_node, var, line_num))
        else:
            for node in body_last_nodes:
                cfg.add_edge(node, condition_node, edge_type='control')

        # The loop can exit after checking the condition
        return [condition_node]

    elif isinstance(statement, DoStatement):
        # Handle do-while loop
        body_node = f'DoWhileBody'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(body_node, label='DoWhileBody', line=line_num, node_type='control')
        cfg.add_edge(current_node, body_node, edge_type='control')

        # Process the body
        body_last_nodes = process_statement(statement.body, cfg, body_node, source_lines, variable_definitions, variable_uses)

        # Process the condition
        condition_node = f'DoWhile({statement.condition})'
        cfg.add_node(condition_node, label=f'DoWhile({statement.condition})', line=line_num, node_type='control')
        for node in body_last_nodes:
            cfg.add_edge(node, condition_node, edge_type='control')

        # Extract variables from condition
        def_vars, use_vars = extract_variables_from_expression(statement.condition)
        for var in def_vars:
            variable_definitions[var] = (condition_node, line_num)
        for var in use_vars:
            variable_uses.append((condition_node, var, line_num))

        # Loop back to body
        cfg.add_edge(condition_node, body_node, edge_type='control')

        # Exit after condition
        return [condition_node]

    elif isinstance(statement, ReturnStatement):
        # Return statement
        return_node = f'Return({statement.expression})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(return_node, label=return_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, return_node, edge_type='control')
        
        # Extract variables from return expression
        if statement.expression:
            def_vars, use_vars = extract_variables_from_expression(statement.expression)
            for var in def_vars:
                variable_definitions[var] = (return_node, line_num)
            for var in use_vars:
                variable_uses.append((return_node, var, line_num))
        
        # Control flow ends here
        return []

    elif isinstance(statement, ThrowStatement):
        # Throw statement
        throw_node = f'Throw({statement.expression})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(throw_node, label=throw_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, throw_node, edge_type='control')
        
        # Extract variables from throw expression
        if statement.expression:
            def_vars, use_vars = extract_variables_from_expression(statement.expression)
            for var in def_vars:
                variable_definitions[var] = (throw_node, line_num)
            for var in use_vars:
                variable_uses.append((throw_node, var, line_num))
        
        # Control flow ends here
        return []

    elif isinstance(statement, BreakStatement):
        # Break statement
        break_node = 'Break'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(break_node, label=break_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, break_node, edge_type='control')
        # Break statements terminate the loop, handled in higher context
        return []

    elif isinstance(statement, ContinueStatement):
        # Continue statement
        continue_node = 'Continue'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(continue_node, label=continue_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, continue_node, edge_type='control')
        # Continue statements loop back, handled in higher context
        return []

    elif isinstance(statement, SynchronizedStatement):
        # Synchronized statement
        sync_node = f'Synchronized({statement.expression})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(sync_node, label=sync_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, sync_node, edge_type='control')
        
        # Extract variables from synchronized expression
        if statement.expression:
            def_vars, use_vars = extract_variables_from_expression(statement.expression)
            for var in def_vars:
                variable_definitions[var] = (sync_node, line_num)
            for var in use_vars:
                variable_uses.append((sync_node, var, line_num))
        
        # Process the block inside synchronized
        return process_block_statements(statement.block, cfg, sync_node, source_lines, variable_definitions, variable_uses)

    elif isinstance(statement, TryStatement):
        # Try statement
        try_node = 'Try'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(try_node, label='Try', line=line_num, node_type='control')
        cfg.add_edge(current_node, try_node, edge_type='control')

        # Process try block
        try_last_nodes = process_block_statements(statement.block, cfg, try_node, source_lines, variable_definitions, variable_uses)

        # Process catches
        catch_last_nodes = []
        for catch_clause in statement.catches:
            catch_node = f'Catch({catch_clause.parameter.name})'
            catch_line = getattr(catch_clause, 'position', None).line if getattr(catch_clause, 'position', None) else None
            cfg.add_node(catch_node, label=catch_node, line=catch_line, node_type='control')
            cfg.add_edge(try_node, catch_node, edge_type='control')
            catch_last_nodes.extend(process_block_statements(catch_clause.block, cfg, catch_node, source_lines, variable_definitions, variable_uses))

        # Process finally block if present
        if statement.finally_block:
            finally_node = 'Finally'
            cfg.add_node(finally_node, label='Finally', line=line_num, node_type='control')
            for node in try_last_nodes + catch_last_nodes:
                cfg.add_edge(node, finally_node, edge_type='control')
            finally_last_nodes = process_block_statements(statement.finally_block, cfg, finally_node, source_lines, variable_definitions, variable_uses)
            return finally_last_nodes
        else:
            # Merge paths from try and catches
            return try_last_nodes + catch_last_nodes

    elif isinstance(statement, SwitchStatement):
        # Handle switch statements
        switch_node = f'Switch({statement.expression})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(switch_node, label=switch_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, switch_node, edge_type='control')
        
        # Extract variables from switch expression
        if statement.expression:
            def_vars, use_vars = extract_variables_from_expression(statement.expression)
            for var in def_vars:
                variable_definitions[var] = (switch_node, line_num)
            for var in use_vars:
                variable_uses.append((switch_node, var, line_num))

        # Process each case
        case_last_nodes = []
        for case in statement.cases:
            case_node = f'Case({case.case})'
            case_line = getattr(case, 'position', None).line if getattr(case, 'position', None) else None
            cfg.add_node(case_node, label=case_node, line=case_line, node_type='control')
            cfg.add_edge(switch_node, case_node, edge_type='control')
            
            # Process statements in this case
            case_body_nodes = process_block_statements(case.statements, cfg, case_node, source_lines, variable_definitions, variable_uses)
            case_last_nodes.extend(case_body_nodes)

        return case_last_nodes

    elif isinstance(statement, StatementExpression):
        # Expression statement
        expr_node = f'{statement.expression}'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(expr_node, label=expr_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, expr_node, edge_type='control')
        
        # Extract variables from expression
        def_vars, use_vars = extract_variables_from_expression(statement.expression)
        for var in def_vars:
            variable_definitions[var] = (expr_node, line_num)
        for var in use_vars:
            variable_uses.append((expr_node, var, line_num))
        
        return [expr_node]

    elif isinstance(statement, LocalVariableDeclaration):
        # Variable declaration
        var_node = f'{statement}'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(var_node, label=var_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, var_node, edge_type='control')
        
        # Extract variables from declaration
        for declarator in statement.declarators:
            if isinstance(declarator, VariableDeclarator):
                # Variable name is defined
                if declarator.name:
                    variable_definitions[declarator.name] = (var_node, line_num)
                # Initializer may use variables
                if declarator.initializer:
                    def_vars, use_vars = extract_variables_from_expression(declarator.initializer)
                    for var in def_vars:
                        variable_definitions[var] = (var_node, line_num)
                    for var in use_vars:
                        variable_uses.append((var_node, var, line_num))
        
        return [var_node]

    elif statement is None or str(statement).strip() == '':
        # Empty statement
        return [current_node]

    else:
        # Other statements
        print(f"Warning: Unknown statement type: {type(statement).__name__} - {statement}")
        stmt_node = f'Unknown({type(statement).__name__})'
        line_num = getattr(statement, 'position', None).line if getattr(statement, 'position', None) else None
        cfg.add_node(stmt_node, label=stmt_node, line=line_num, node_type='control')
        cfg.add_edge(current_node, stmt_node, edge_type='control')
        return [stmt_node]

def generate_control_flow_graphs(java_file_path, output_base_dir='cfg_output'):
    """
    Generates control flow graphs for a given Java file.
    """
    print(f"DEBUG: Parsing Java file: {java_file_path}")
    parsed_java_code = parse_java_file(java_file_path)
    if parsed_java_code is None:
        print(f"DEBUG: Failed to parse Java file: {java_file_path}")
        # Fallback: emit a per-line linear CFG with dataflow via identifier co-occurrence
        with open(java_file_path, 'r') as f:
            lines = f.readlines()
        cfg = nx.MultiDiGraph()
        prev = None
        last_line_by_ident = {}
        import re
        ident = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        for i, line in enumerate(lines, start=1):
            node = f"L{i}"
            cfg.add_node(node, label=line.strip(), line=i, node_type='stmt')
            if prev is not None:
                cfg.add_edge(prev, node, edge_type='control')
            prev = node
            seen = set()
            for m in ident.finditer(line):
                name = m.group(0)
                if name in ("true","false","null"): continue
                if name in seen: continue
                seen.add(name)
                if name in last_line_by_ident:
                    cfg.add_edge(last_line_by_ident[name], node, edge_type='dataflow', variable=name)
                last_line_by_ident[name] = node
        print(f"DEBUG: Fallback linear CFG with {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")
        return [(os.path.splitext(os.path.basename(java_file_path))[0], cfg)]
    
    print(f"DEBUG: Successfully parsed Java file")
    with open(java_file_path, 'r') as f:
        source_lines = f.readlines()
    print(f"DEBUG: Read {len(source_lines)} source lines")
    
    method_declarations = extract_method_declarations(parsed_java_code)
    print(f"DEBUG: Found {len(method_declarations)} method declarations")
    
    cfgs = []
    for i, method in enumerate(method_declarations):
        print(f"DEBUG: Processing method {i+1}/{len(method_declarations)}: {method.name}")
        cfg = create_cfg(method, source_lines)
        print(f"DEBUG: Created CFG for {method.name} with {cfg.number_of_nodes()} nodes, {cfg.number_of_edges()} edges")
        cfgs.append((method.name, cfg))

    print(f"DEBUG: Generated {len(cfgs)} CFGs total")
    return cfgs

def save_cfgs(cfgs, output_dir='cfg_output'):
    """
    Saves the CFGs for each method into JSON files for machine learning models.
    Now includes dataflow information with different edge types.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for method_name, cfg in cfgs:
        # Prepare data for JSON serialization
        data = {
            'method_name': method_name,
            'java_file': None,  # Will be set by caller if needed
            'nodes': [],
            'edges': [],
            'control_edges': [],
            'dataflow_edges': []
        }
        node_id_map = {}  # Map node names to unique IDs

        for idx, (node, attr) in enumerate(cfg.nodes(data=True)):
            node_id_map[node] = idx
            data['nodes'].append({
                'id': idx,
                'label': attr.get('label', ''),
                'line': attr.get('line', None),
                'node_type': attr.get('node_type', 'control')
            })

        # Separate control flow and dataflow edges
        for source, target, edge_attr in cfg.edges(data=True):
            edge_data = {
                'source': node_id_map[source],
                'target': node_id_map[target]
            }
            
            edge_type = edge_attr.get('edge_type', 'control')
            if edge_type == 'dataflow':
                edge_data['variable'] = edge_attr.get('variable', '')
                data['dataflow_edges'].append(edge_data)
            else:
                data['control_edges'].append(edge_data)
            
            # Also add to general edges list for backward compatibility
            data['edges'].append(edge_data)

        # Save to JSON file
        file_name = f'{method_name}.json'
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'CFG with dataflow saved for method "{method_name}" at {file_path}')

# Example usage
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--java_file', required=True)
    parser.add_argument('--out_dir', default='cfg_output')
    args = parser.parse_args()
    
    print(f"=== CFG GENERATION DEBUG ===")
    print(f"Input Java file: {args.java_file}")
    print(f"Output directory: {args.out_dir}")
    print(f"File exists: {os.path.exists(args.java_file)}")
    
    if not os.path.exists(args.java_file):
        print(f"ERROR: Java file does not exist: {args.java_file}")
        exit(1)
    
    print(f"Starting CFG generation...")
    cfgs = generate_control_flow_graphs(args.java_file, args.out_dir)
    print(f"Generated {len(cfgs)} CFGs")
    
    # Save under a per-file subdirectory using the base name
    base = os.path.splitext(os.path.basename(args.java_file))[0]
    out_dir = os.path.join(args.out_dir, base)
    print(f"Saving CFGs to: {out_dir}")
    
    save_cfgs(cfgs, out_dir)
    print(f"CFG generation completed successfully!")
