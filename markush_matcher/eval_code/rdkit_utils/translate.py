import re
import json
import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Tuple, Union, List, Dict, Any

from rdkit import Chem, RDLogger

from . import misc
from .misc import Index, AtomIndex, RingIndex


logger = logging.getLogger(__name__)

@unique
class TextType(Enum):
    SYMBOL = 'symbol'
    SCRIPT = 'script'
    MULTIPLE = 'multiple'
    PRIME = 'prime'


class Tokens:
    atom_start = '<a>'
    atom_end = '</a>'
    circ_start = '<c>'
    circ_end = '</c>'
    ring_start = '<r>'
    ring_end = '</r>'
    dummy = '<dum>'
    separator = '<sep>'
    grp_content = re.compile(
        rf'(?P<{TextType.SYMBOL.value}>[A-Za-z0-9]*)' +
        rf'(?P<{TextType.SCRIPT.value}>(\[\S+\])?)' +
        rf'(?P<{TextType.PRIME.value}>[\'\"]?)' +
        rf'(?P<{TextType.MULTIPLE.value}>(\?([a-z]|\d{1}|\d-\d)$)?)'  # $ for matching string ending
    )
    grp_pattern = re.compile(
        rf'({atom_start}|{circ_start}|{ring_start}|{ring_start}{circ_start})' +
        rf'(\d+:\S+?)' +
        rf'({atom_end}|{circ_end}|{ring_end})'
    )


@dataclass
class GroupDesc:
    id: Index
    symbol: Optional[str] = None
    script: Optional[str] = None
    prime: Optional[str] = None
    multiple: Optional[str] = None
    is_circle: bool = False
    is_dummy: bool = False
    
    @property
    def is_abbrev(self) -> bool:
        """Determine if the group describes abbreviation."""
        return (
            isinstance(self.id, AtomIndex)
            and (not self.is_dummy)
            and (not self.is_circle)
            and misc.is_abbrev(self.symbol)
            and self.script is None
            and self.prime is None
            and self.multiple is None
        )
        
    def __str__(self) -> str:
        """A simplified stringified notation for group name."""
        if self.is_dummy:
            return Tokens.dummy
        # XXX Ignore `multiple` for now.
        return ''.join([self.symbol or '', self.script or '', self.prime or ''])
            
    def to_json(self) -> str:
        # TODO virtual ring substitute
        if isinstance(self.id, RingIndex) and self.id.virtual:
            return ''
        
        items = []
        idx_type = 'atom' if isinstance(self.id, AtomIndex) else 'ring'
        items.append(f'"idx":{self.id}')
        items.append(f'"idxType":"{idx_type}"')
        
        def assemble(items_: List[str]) -> str:
            return '{' + ','.join(items_) + '}'
        
        if self.is_dummy:
            # Special case: attachment point.
            items.append(f'"symbol":"{Tokens.dummy}"')
            return assemble(items)
        
        # TODO circled abstract ring
        if self.is_circle:
            items.append(f'"symbol":"{self.symbol}"')
            return assemble(items)
        
        if self.symbol is not None:
            items.append(f'"symbol":"{self.symbol}"')
        if self.script is not None:
            items.append(f'"script":"{self.script}"')
        if self.prime is not None:
            items.append(f'"prime":"{self.prime}"')
        if self.multiple is not None:
            items.append(f'"multiple":"{self.multiple}"')
        return assemble(items)


class Translator:
    """A helper class for converting structures between molecule recognizer and editor."""
    @classmethod
    def parse_caption(
        cls, 
        caption: str, 
        return_mol: bool = False,
        error_msg: bool = False
    ) -> Optional[Tuple[Union[Chem.rdchem.Mol, str], str]]:
        """Parse a complete molecule caption or prediction."""
        smi, *groups = caption.split(Tokens.separator)
        if len(groups) != 1:
            if error_msg:
                logger.warning(
                    f'{len(groups)} `{Tokens.separator}` found in caption: {caption}'
                )
            return
    
        if error_msg:
            RDLogger.EnableLog('rdApp.*')
        
        mol = Chem.MolFromSmiles(smi)
        RDLogger.DisableLog('rdApp.*')
        if mol is None:
            if error_msg:
                logger.warning(f'Invalid SMILES: {smi}')
            return
        if return_mol:
            return mol, groups[0]
        return smi, groups[0]
    
    @classmethod
    def parse_groups(cls, seq: str) -> List[GroupDesc]:
        """Parse R group texts from predictions."""
        if seq == '':
            return []
        descriptions = []
        for grp_start, grp_content, _ in re.findall(Tokens.grp_pattern, seq):
            parsed = cls.parse_group(grp_content)
            if parsed is None:
                continue
            idx, grp_text = parsed
            if grp_start == Tokens.atom_start:
                grp_desc = GroupDesc(id=AtomIndex(idx))
                if len(grp_text) == 0:
                    grp_desc.is_dummy = True
            elif grp_start == Tokens.circ_start:
                grp_desc = GroupDesc(id=AtomIndex(idx), is_circle=True)
            elif grp_start == f'{Tokens.ring_start}{Tokens.circ_start}':
                grp_desc = GroupDesc(id=RingIndex(idx, virtual=True))
            elif grp_start == Tokens.ring_start:
                grp_desc = GroupDesc(id=RingIndex(idx, ring=True))
            else:
                continue
            grp_desc.symbol = grp_text.get(TextType.SYMBOL)
            grp_desc.script = grp_text.get(TextType.SCRIPT)
            grp_desc.prime = grp_text.get(TextType.PRIME)
            grp_desc.multiple = grp_text.get(TextType.MULTIPLE)
            descriptions.append(grp_desc)
        return descriptions
    
    @classmethod
    def parse_group(cls, group: str) -> Optional[Tuple[int, Dict[TextType, str]]]:
        """Get text of a R group."""
        items = group.split(':')
        if len(items) != 2:
            return
        idx, content = items
        if not idx.isdigit():
            return
        idx = int(idx)
        if content == Tokens.dummy:
            return idx, {}
        # Gotta make sure that not all text types are empty string.
        grp_text = cls.get_group_texts(content)
        if len(grp_text) == 0:
            return
        return idx, grp_text
        
    @classmethod
    def get_group_texts(cls, content: str) -> Dict[TextType, str]:
        """Helper for reorganizing texts in R group."""
        texts = {}
        # NOTE Unmatched items will be empty strings.
        matched = re.match(Tokens.grp_content, content).groupdict()
        for tt_value, text in matched.items():
            if len(text) == 0:
                continue
            if tt_value == TextType.SYMBOL.value:
                texts[TextType.SYMBOL] = text
            elif tt_value == TextType.SCRIPT.value:
                # Remove square brackets.
                assert text.startswith('[') and text.endswith(']')
                texts[TextType.SCRIPT] = text[1:-1]
            elif tt_value == TextType.PRIME.value:
                texts[TextType.PRIME] = text
            elif tt_value == TextType.MULTIPLE.value:
                assert text.startswith('?')
                texts[TextType.MULTIPLE] = text[1:]
        return texts
    

def _remove_atoms(rwmol, to_remove: List[int]) -> bool:
    """Remove dummy atoms for abbrevs altogether in reverse order, this will 
       not affect newly added atoms.
    """
    try:
        for i in sorted(to_remove, reverse=True):
            rwmol.RemoveAtom(i)
        Chem.SanitizeMol(rwmol)  # NOTE must do
        return True
    except Exception as e:
        logger.error(str(e))
        return False


def preprocess_caption(caption: str) -> Optional[Tuple[str, str]]:
    """Parse molecule recognition caption and prepare input structure for editor
       as well as SMILES for display.
    """
    # Skip invalid predictions.
    if caption is None or caption == '': 
        return 
    parsed = Translator.parse_caption(caption, return_mol=True)
    if parsed is None:
        return
    
    mol, groups = parsed
    grp_descriptions = Translator.parse_groups(groups)
    num_atoms = mol.GetNumAtoms()
    num_rings = mol.GetRingInfo().NumRings()
    molecule = misc.to_cxsmiles(mol) + Tokens.separator
    
    # Append extra group information if necessary.
    # Substitute abbrevs and R-group labels to get SMILES for display.
    extra = []
    rwmol = Chem.RWMol(mol)
    atom_to_remove: List[int] = []
    for desc in grp_descriptions:
        if not misc.is_valid_index(desc.id, num_atoms, num_rings):
            continue
        # Append extra group info in JSON.
        grp_json = desc.to_json()
        if isinstance(grp_json, str) and len(grp_json) > 0:
            extra.append(grp_json)
        # Substitute abbrevs and R-group labels for displaying SMILES.
        if not isinstance(desc.id, AtomIndex):
            continue
        i = int(desc.id)
        atom = rwmol.GetAtomWithIdx(i)
        if atom.GetSymbol() != '*':
            continue
        grp_smi = misc.get_abbrev_smi(desc.symbol, atom.GetDegree())
        if (
            desc.is_abbrev 
            and misc.is_atom_valid_for_abbrev(atom)
            and grp_smi is not None
        ):
            # Functional group abbreviation.
            try:
                misc.merge_abbrev_group(tgt_mol=rwmol, src=grp_smi, attach_idx=i)
                atom_to_remove.append(i)
            except Exception as e:
                logger.error(str(e))
                continue
        elif desc.is_dummy:
            # Attachment point.
            atom.SetProp('group', 'AP')
        elif desc.symbol is not None:
            # R-group label.
            label = desc.symbol
            label += '' if desc.script is None else desc.script
            label += '' if desc.prime is None else desc.prime
            if desc.multiple is not None:
                label += f'?{desc.multiple}'
            atom.SetProp('group', label)

    removal_success = _remove_atoms(rwmol, atom_to_remove)
    if not removal_success:
        return

    if len(extra) > 0:  
        molecule += '{"extra":[' + ','.join(extra) + ']}'
        
    return misc.get_smiles_for_display(rwmol.GetMol()), molecule


def _strip_cxsmiles(cxsmi: str) -> Optional[str]:
    # A valid CXSMILES: '*c1nc(O)c2cc(Cl)ncc2n1 |$*;;;;;;;;;;;;$|'
    if cxsmi == '':  # deal with initial callback
        return
    items = cxsmi.split(' ')
    if len(items) > 2:
        logger.error(f'Invalid SMILES or CXSMILES: {cxsmi}')
        return
    return items[0]


def _load_extra(obj: str) -> Optional[List[Dict[str, Any]]]:
    # TODO problem occurs when `prime` == '"' (double quote)
    try:
        return json.loads(obj)['extra']  # List[Dict]
    except:
        if len(obj) > 0:
            logger.error(f'Invalid extra information: {obj}')
        return
    
    
def _get_mapped_groups(
    groups: List[Dict[str, Any]],
    atom_mapping: Dict[int, int],
    ring_mapping: Optional[Dict[int, int]] = None
) -> List[Dict[str, Any]]:
    new_groups = []
    for grp in groups:
        idx_type = grp.get('idxType')
        if grp.get('idx') is None or idx_type not in ('atom', 'ring'):
            continue
        mapping = {}
        if idx_type == 'ring' and ring_mapping is not None:
            mapping = ring_mapping
        elif idx_type == 'atom':
            mapping = atom_mapping
        new_idx = mapping.get(grp['idx'])
        if new_idx is not None:
            grp['idx'] = new_idx
            new_groups.append(grp)
            
    return new_groups


def postprocess_struct(struct: str) -> Optional[Tuple[str, str]]:
    """Parse molecule structure yielded from the editor into SMILES and extra
       representations.
    """
    if struct is None or struct == '':
        return
    items = struct.split(Tokens.separator)
    if len(items) != 3:
        logger.error(f'Invalid struct from editor: {struct}')
        return

    # 1) `cxsmi`: SMILES or CXSMILES NOTE atoms might be reordered after editing!
    # 2) `extra`: extra group info in JSON which keeps original atom order
    # 3) `block`: molecule structure block which keeps original atom order
    cxsmi, extra, block = items
    smi = _strip_cxsmiles(cxsmi)
    if smi is None:
        return
        
    # NOTE Do not touch parsed SMILES even if it is not canonical.
    m1 = Chem.MolFromSmiles(smi)
    m2 = Chem.MolFromMolBlock(block)
    if m1.GetNumAtoms() != m2.GetNumAtoms():
        logger.error(f'Atom count mismatch: {m1.GetNumAtoms()} and {m2.GetNumAtoms()}')
        return
    
    match = m1.GetSubstructMatch(m2)  # Tuple[int]
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol.GetSubstructMatch
    # NOTE `match` holds the atom mapping between `m1` and `m2`, for example:
    # match = (0, 12, 2, 11, 10, 9, 7, 8, 5, 6, 3, 4, 1)
    # Then the atom mapping can be deduced as below
    # m1 | 0 | 12 | 2 | 11 | 10 | 9 | 7 | 8 | 5 | 6 | 3  | 4  | 1  |
    # m2 | 0 | 1  | 2 | 3  | 4  | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
    if len(match) == 0:
        logger.error(f'Atom mismatch between {smi} and {block}')
        return
    
    atom_mapping = {i: j for i, j in enumerate(match)}  # from `m2` to `m1`
    inv_atom_mapping = {j: i for i, j in enumerate(match)}  # from `m1` to `m2`
    
    # As `m1` holds the proper atom order that can be written in SMILES,
    # we convert atom & ring index in `extra` to match `m1`.
    cxsmi_suffix = ['*' if a.GetSymbol() == '*' else '' for a in m1.GetAtoms()]
    molecule = smi + f' |${";".join(cxsmi_suffix)}$|' + Tokens.separator
    
    groups = _load_extra(extra)
    new_groups = []
    if groups is not None:
        ring_mapping = misc.get_ring_mapping(tgt_mol=m1, atom_mapping=inv_atom_mapping)
        new_groups = _get_mapped_groups(groups, atom_mapping, ring_mapping)
        
    # Add extra group information.
    if len(new_groups) > 0:
        molecule += json.dumps({'extra': new_groups}, separators=(',', ':'))
    
    # Now group index matches the new molecule, we can substitute functional
    # group abbrevs and R-groups back into it to get SMILES only for display
    # purpose. NOTE This step will not affect `molecule`, which stores the actual
    # structure for editing and modelling.
    rwmol = Chem.RWMol(m1)
    atom_to_remove: List[int] = []
    for grp in new_groups:
        if grp.get('idxType') != 'atom':
            continue
        i = grp.get('idx', -1)
        if not 0 <= i < rwmol.GetNumAtoms():
            continue
        atom = rwmol.GetAtomWithIdx(i)
        symbol = grp.get('symbol')
        if atom.GetSymbol() != '*' or symbol is None:
            continue
        grp_smi = misc.get_abbrev_smi(symbol, atom.GetDegree())
        if (
            misc.is_abbrev(symbol) 
            and misc.is_atom_valid_for_abbrev(atom)
            and grp_smi is not None
        ):
            # Functional group abbreviation.
            try:
                misc.merge_abbrev_group(tgt_mol=rwmol, src=grp_smi, attach_idx=i)
                atom_to_remove.append(i)
            except Exception as e:
                logger.error(str(e))
                continue
        elif symbol == Tokens.dummy:
            # Dummy attachment point.
            atom.SetProp('group', 'AP')
        else: 
            # R-group label.
            label = symbol + grp.get('script', '') + grp.get('prime', '')
            multiple = grp.get('multiple')
            if multiple is not None:
                label += f'?{multiple}'
            atom.SetProp('group', label)
    
    removal_success = _remove_atoms(rwmol, atom_to_remove)
    if not removal_success:
        return
    
    return misc.get_smiles_for_display(rwmol.GetMol()), molecule
