'''Unit tests for the package.'''

import pytest
from pdbgraph import parse_pdb_file, construct_torch_graph


@pytest.fixture(scope="module")
def sample_pdb_file(tmpdir_factory):
    """
    Returns:
    str: The path to the sample PDB file.
    """
    return "tests/test.pdb"


def test_parse_pdb_file(sample_pdb_file):
    """Test the parse_pdb_file function."""
    atoms, bonds = parse_pdb_file(sample_pdb_file)
    # Perform assertions to verify the output
    assert len(atoms) == 114  # Check the number of atoms
    assert len(bonds) == 208  # Check the number of bonds


def test_construct_torch_graph():
    """Test the construct_torch_graph function."""
    # Define sample atoms and bonds data
    sample_atoms = [
        {'coord': [3.373, -4.042, 1.886]},  # Atom 1
        {'coord': [4.658, -4.682, 1.669]},  # Atom 2
        {'coord': [5.671, -3.727, 2.302]},  # Atom 3
        {'coord': [5.285, -2.545, 2.254]},  # Atom 4
        {'coord': [4.476, -5.989, 0.862]},  # Atom 5
        {'coord': [5.654, -6.823, 1.350]},  # Atom 6
        {'coord': [6.593, -6.258, 2.003]},  # Atom 7
        {'coord': [5.682, -8.077, 1.067]},  # Atom 8
        {'coord': [6.882, -4.204, 2.815]},  # Atom 9
        {'coord': [7.996, -3.408, 3.383]},  # Atom 10
        {'coord': [7.674, -2.036, 3.945]},  # Atom 11
        {'coord': [6.568, -1.614, 4.302]},  # Atom 12
        {'coord': [9.194, -4.300, 3.746]},  # Atom 13
        {'coord': [10.350, -3.452, 4.256]},  # Atom 14
        {'coord': [9.591, -5.645, 3.142]}   # Atom 15
    ]

    sample_bonds = [
        {
            'atom1_coord': [3.373, -4.042, 1.886],
            'atom2_coord': [4.658, -4.682, 1.669]
        },  # Bond 1
        {
            'atom1_coord': [4.658, -4.682, 1.669],
            'atom2_coord': [5.671, -3.727, 2.302]
        },  # Bond 2
        {
            'atom1_coord': [5.671, -3.727, 2.302],
            'atom2_coord': [5.285, -2.545, 2.254]
        },  # Bond 3
        {
            'atom1_coord': [4.658, -4.682, 1.669],
            'atom2_coord': [4.476, -5.989, 0.862]
        },  # Bond 4
        {
            'atom1_coord': [4.476, -5.989, 0.862],
            'atom2_coord': [5.654, -6.823, 1.350]
        },  # Bond 5
        {
            'atom1_coord': [5.654, -6.823, 1.350],
            'atom2_coord': [6.593, -6.258, 2.003]
        },  # Bond 6
        {
            'atom1_coord': [5.654, -6.823, 1.350],
            'atom2_coord': [5.682, -8.077, 1.067]
        },  # Bond 7
        {
            'atom1_coord': [6.882, -4.204, 2.815],
            'atom2_coord': [7.996, -3.408, 3.383]
        },  # Bond 8
        {
            'atom1_coord': [7.996, -3.408, 3.383],
            'atom2_coord': [7.674, -2.036, 3.945]
        },  # Bond 9
        {
            'atom1_coord': [7.674, -2.036, 3.945],
            'atom2_coord': [6.568, -1.614, 4.302]
        },  # Bond 10
        {
            'atom1_coord': [7.996, -3.408, 3.383],
            'atom2_coord': [9.194, -4.300, 3.746]
        },  # Bond 11
        {
            'atom1_coord': [9.194, -4.300, 3.746],
            'atom2_coord': [10.350, -3.452, 4.256]
        },  # Bond 12
        {
            'atom1_coord': [9.194, -4.300, 3.746],
            'atom2_coord': [9.591, -5.645, 3.142]
        }    # Bond 13
    ]

    # Call the function
    data = construct_torch_graph(sample_atoms, sample_bonds)
    # Perform assertions to verify the output
    assert data.x.shape[0] == len(sample_atoms)  # number of nodes
    assert data.edge_index.shape[1] == len(sample_bonds)  # number of edges
