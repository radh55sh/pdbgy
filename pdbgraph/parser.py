'''Module for parsing PDB files and extracting atom and bond information.'''

from Bio.PDB import PDBParser


def parse_pdb_file(pdb_file_path):
    """
    Parse a PDB file and extract atom and bond information.

    Args:
        pdb_file_path (str): Path to the PDB file.

    Returns:
        tuple: A tuple containing lists of atoms and bonds.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)
    atoms = []
    bonds = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Store atom information
                    atom_info = {
                        'id': atom.get_id(),
                        'name': atom.get_name(),
                        'coord': atom.get_coord(),
                        }
                    atoms.append(atom_info)

                    # Find bonds between atoms
                    for neighbor in atom.get_parent().get_atoms():
                        if atom != neighbor and atom - neighbor < 2.0:
                            # Store bond information
                            bond_info = {
                                'atom1_id': atom.get_id(),
                                'atom1_coord': atom.get_coord(),
                                'atom2_id': neighbor.get_id(),
                                'atom2_coord': neighbor.get_coord(),
                                'distance': atom - neighbor,
                                }
                            bonds.append(bond_info)

    return atoms, bonds
