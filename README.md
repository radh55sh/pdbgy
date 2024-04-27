# pdbgraph

The `pdbgraph` package allows you to convert PDB (Protein Data Bank) files into graph data, which can be used for various purposes, including graph neural networks (GNNs) and protein structure analysis.

## Features

- Convert PDB files into graph data.
- Directly output graph data for further processing.
- Additional functionalities will be added to facilitate the construction of GNN encoders from PDB files.

## Installation

You can install `pdbgraph` via pip:


## Usage

```python
from pdbgraph.parser import parse_pdb_file
from pdbgraph.graph import construct_torch_graph

# Parse a PDB file and extract atom and bond information
atoms, bonds = parse_pdb_file("path/to/your/pdb/file.pdb")

# Construct a torch_geometric graph from the parsed atom and bond information
graph_data = construct_torch_graph(atoms, bonds)

# Now you can use graph_data for further processing or training GNNs
