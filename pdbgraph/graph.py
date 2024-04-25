import torch
from torch_geometric.data import Data


def construct_torch_graph(atoms, bonds):
    """
    Construct a torch geometric graph from atom and bond information.

    Args:
        atoms (list): List of dictionaries containing atom information.
        bonds (list): List of dictionaries containing bond information.

    Returns:
        Data: A torch geometric Data object representing the graph.
    """
    # Create node features (coordinates) tensor
    x = torch.tensor([atom['coord'] for atom in atoms], dtype=torch.float)
    # Create edge_index tensor
    edge_index = []
    for bond in bonds:
        atom1_index = next
        (
            i for i, atom_info in enumerate(atoms)
            if tuple(atom_info['coord']) == tuple(bond['atom1_coord'])
        )
        atom2_index = next
        (
            i for i, atom_info in enumerate(atoms)
            if tuple(atom_info['coord']) == tuple(bond['atom2_coord'])
        )

        atom1_index = next
        (
            i for i, atom_info in enumerate(atoms)
            if tuple(atom_info['coord']) == tuple(bond['atom1_coord'])
        )
        atom2_index = next
        (
            i for i, atom_info in enumerate(atoms)
            if tuple(atom_info['coord']) == tuple(bond['atom2_coord'])
        )
        edge_index.append([atom1_index, atom2_index])

    # Convert edge_index to PyTorch LongTensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Create Data object
    data = Data(x=x, edge_index=edge_index)

    return data
