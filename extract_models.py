import ast

def extract_classes(filepath, target_classes, out_filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    extracted_nodes = []
    
    # We also want to extract standard imports and helper functions at the top of the file
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            extracted_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ['init_layer', 'init_bn', 'drop_path']:
            extracted_nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in target_classes:
            extracted_nodes.append(node)
            
    with open(out_filepath, 'w') as f:
        for node in extracted_nodes:
            f.write(ast.unparse(node))
            f.write("\n\n")

if __name__ == "__main__":
    target = ['ConvBlock', 'ConvBlock5x5', 'Cnn6', 'Cnn10', 'MobileNetV1', 'MobileNetV2', 'ConvBlockPW', 'ConvBlockDW']
    extract_classes('audioset_tagging_cnn/pytorch/models.py', target, 'internal_models.py')
