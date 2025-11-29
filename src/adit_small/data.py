
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math, random

try:
    from torch_geometric.datasets import QM9
    HAS_PYG = True
except Exception:
    HAS_PYG = False

ATOM_TYPES = ["H","C","N","O","F"]  # QM9 subset

def to_item(graph):
    # graph.z: atomic numbers, graph.pos: (N,3)
    z2idx = {1:0, 6:1, 7:2, 8:3, 9:4}
    types = torch.tensor([z2idx.get(int(a), 1) for a in graph.z], dtype=torch.long)
    coords = graph.pos.float()
    return {"types": types, "coords": coords}

class QM9Wrapper(Dataset):
    def __init__(self, ds, augment=True):
        self.ds = ds
        self.augment = augment
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        item = to_item(self.ds[i])
        if self.augment:
            R = random_rotation_matrix()
            t = torch.randn(3) * 0.05
            item["coords"] = (item["coords"] @ R.T) + t
        return item

def random_rotation_matrix():
    # Uniform random rotation via Euler angles
    a,b,c = [random.random()*2*math.pi for _ in range(3)]
    Rx = torch.tensor([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    Ry = torch.tensor([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]])
    Rz = torch.tensor([[math.cos(c),-math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
    return (Rz@Ry@Rx).float()

class ToyMoleculeSet(Dataset):
    """Fallback tiny dataset if PyG/QM9 isn't available."""
    def __init__(self, n=200, max_atoms=9):
        self.n=n; self.max_atoms=max_atoms
    def __len__(self): return self.n
    def __getitem__(self, idx):
        n_atoms = random.randint(5, self.max_atoms)
        types = torch.randint(low=0, high=len(ATOM_TYPES), size=(n_atoms,))
        coords = torch.randn(n_atoms,3)*0.7
        return {"types": types, "coords": coords}

def pad_batch(batch, pad_value=-1):
    max_n = max(item["types"].size(0) for item in batch)
    B = len(batch)
    types = torch.full((B,max_n), pad_value, dtype=torch.long)
    coords = torch.zeros(B,max_n,3)
    mask = torch.zeros(B,max_n, dtype=torch.bool)
    for i,item in enumerate(batch):
        n=item["types"].size(0)
        types[i,:n]=item["types"]
        coords[i,:n]=item["coords"]
        mask[i,:n]=True
    return {"types":types, "coords":coords, "mask":mask}

def get_qm9_dataloaders(root:str, batch_size:int=64, num_workers:int=2, augment:bool=True)->Tuple[DataLoader,DataLoader]:
    if HAS_PYG:
        ds = QM9(root)
        ds_w = QM9Wrapper(ds)
        # 🔍 Debug: show dataset choice + size
        print("Using dataset: QM9")
        print("Dataset length:", len(ds_w))

        n = len(ds_w)
        n_train = int(0.95*n)
        train, val = random_split(ds_w, [n_train, n-n_train])
        dl_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_batch, pin_memory=True)
        dl_val   = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_batch, pin_memory=True)
        return dl_train, dl_val
    else:
        ds = ToyMoleculeSet(n=800)

        # 🔍 Debug: show dataset choice + size
        print("Using dataset: Toy synthetic")
        print("Dataset length:", len(ds_w))

        n_train = int(0.9*len(ds))
        train, val = random_split(ds, [n_train, len(ds)-n_train])
        dl_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_batch)
        dl_val   = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_batch)
        return dl_train, dl_val
