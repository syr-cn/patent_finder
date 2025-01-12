from rdkit_utils.misc import AtomIndex, RingIndex

def mol_generation(caption,r_groups=None):
    '''
    extend MoleculeQuerier and generate molecule with LLM response group
    '''
    parsed = Translator.parse_caption(caption, return_mol=True)
    if parsed is None:
        raise ValueError(f'Incorrect caption: {caption}')
    mol, groups = parsed
    grp_descriptions = Translator.parse_groups(groups)
    atom_list = []
    ring_list = []
    # import pdb; pdb.set_trace()
    '''add ring judge. if has ring subsitute like <r>x</r>, 
    make the mol to a group of mol where the ring subsitute is replaced by atom subsitute
    '''
    for grp in grp_descriptions:
        if isinstance(grp.id, RingIndex):
            ring_list.append(grp)
        elif isinstance(grp.id, AtomIndex):
            atom_list.append(grp)
    if len(ring_list) == 0:
        mol_querier = MoleculeQuerier(mol, atom_list)
        mols = mol_querier._replace_placeholders(r_groups,caption=caption)
        
        return mols
    return []