from typing import Union


class Index(int):
    def __new__(cls, value: int, **kwds):
        assert isinstance(value, int) and value >= 0
        return super().__new__(cls, value)


class AtomIndex(Index):
    key: int = 1

    def __hash__(self):
        return hash(('atom', int(self)))

    def __eq__(self, other):
        if isinstance(other, AtomIndex):
            return hash(self) == hash(other)
        return False


class RingIndex(Index):
    # add 'ring' attribute to distinguish from AtomIndex
    def __init__(self, value: int, virtual: bool = False, ring: bool = False) -> None:
        self._virtual = virtual
        self._key = 2_000_000 if self._virtual else 1_000_000
        self._ring = ring
    def __hash__(self):
        return hash(('ring', int(self), self._virtual))

    def __eq__(self, other):
        if isinstance(other, RingIndex):
            return hash(self) == hash(other)
        return False

    @property
    def virtual(self) -> bool:
        return self._virtual

    @property
    def key(self) -> bool:
        return self._key

    @property
    def ring(self) -> bool:
        return self._ring

def is_valid_index(
    idx: Union[AtomIndex, RingIndex], 
    num_atoms: int,
    num_rings: int
) -> bool:
    """Check if `AtomIndex` or `RingIndex` is in range."""
    if isinstance(idx, AtomIndex):
        return 0 <= int(idx) < num_atoms
    elif isinstance(idx, RingIndex):
        if idx.virtual:
            return False
        return 0 <= int(idx) < num_rings
    else:
        return False