from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class ZAtomType:
    """获取特征"""
    
    def __init__(self, mol):
        self.mol = mol

    def read_sdf(self, file_path):
        # Readtwo SDF files with rdkit
        self.mol = Chem.SDMolSupplier(file_path)[0]

    def get_atom_ring_types(self):
        # 获取每个原子的环类型，包括8种类型：
        # {0: 非环, 1: 3元环, 2: 4元环, 3: 5元环, 4: 6元环, 5: 5&6元环, 6: 6&6元环, 7: 其他环}
        ring_info = self.mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # get all atoms in ring
        dict_atom_ringsize = dict()
        for r in atom_rings:
            for i in r:
                if i in dict_atom_ringsize:
                    dict_atom_ringsize[i].append(len(r))
                else:
                    dict_atom_ringsize[i] = [len(r)]

        atom_ring_types = []
        # 获取所有环，按原子索引
        for i in range(self.mol.GetNumAtoms()):
            at = 7

            if i not in dict_atom_ringsize:
                at = 0
            
            elif len(dict_atom_ringsize[i]) == 1:
                if ring_info.IsAtomInRingOfSize(i, 3):
                    at = 1
                elif ring_info.IsAtomInRingOfSize(i, 4):
                    at = 2
                elif ring_info.IsAtomInRingOfSize(i, 5):
                    at = 3
                elif ring_info.IsAtomInRingOfSize(i, 6):
                    at = 4
            
            elif len(dict_atom_ringsize[i]) == 2:
                if set(dict_atom_ringsize[i]) == {5, 6}:
                    at = 5
                elif set(dict_atom_ringsize[i]) == {6}:
                    at = 6

        
            atom_ring_types.append(at)

        return atom_ring_types


    def get_atom_charge_types(self, ff=False):       
        atom_charges = []

        if ff: # MMFF94
            # 生成初始构象
            AllChem.EmbedMolecule(self.mol)  

            # 优化分子构象
            AllChem.MMFFOptimizeMolecule(self.mol)

            # 计算 MMFF94 电荷
            mp = AllChem.MMFFGetMoleculeProperties(self.mol)
            for atom in self.mol.GetAtoms():
                charge = mp.GetMMFFPartialCharge(atom.GetIdx())
                atom_charges.append(charge)

        else: # Gasteiger 电荷
            AllChem.ComputeGasteigerCharges(self.mol)

            for atom in self.mol.GetAtoms():
                charge = atom.GetProp('_GasteigerCharge')
                atom_charges.append(charge)

        return [float(a) for a in atom_charges]


    def get_atom_bond_number(self):
        atom_bond_number = [len(atom.GetBonds()) for atom in self.mol.GetAtoms()]
        return atom_bond_number


    def get_atom_element_types(self):
        atom_element_types = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        return atom_element_types
    




class ZScore:
    def __init__(self, mol_q, mol_t):
        # # 补氢
        # self.mol = Chem.AddHs(self.mol)  

        self.at_q = ZAtomType(mol_q)
        self.at_t = ZAtomType(mol_t)

    def score_element(self):
        # 1 if at_q == at_t, 0 otherwise
        # size=m list and size=n list to create a m*n matrix
        # 扩展两个矩阵，生成 m*n 的比较矩阵
        q_types = np.array(self.at_q.get_atom_element_types())
        t_types = np.array(self.at_t.get_atom_element_types())

        return (q_types[:, np.newaxis] == t_types[np.newaxis, :])

    def score_charge(self):
        q_charges = np.array(self.at_q.get_atom_charge_types())
        t_charges = np.array(self.at_t.get_atom_charge_types())
        charges_diff = np.abs(q_charges[:, np.newaxis] - t_charges[np.newaxis, :])
        y = 10 * charges_diff - 1
        
        return np.where(y < 1, 0.25 * (y + 2) * (y - 1) * (y - 1), 0)

    def score_bond(self):
        q_bonds = np.array(self.at_q.get_atom_bond_number())
        t_bonds = np.array(self.at_t.get_atom_bond_number())

        return (q_bonds[:, np.newaxis] == t_bonds[np.newaxis, :])

    def score_ring(self):
        q_rings = np.array(self.at_q.get_atom_ring_types())
        t_rings = np.array(self.at_t.get_atom_ring_types())

        return (q_rings[:, np.newaxis] == t_rings[np.newaxis, :])

    def score(self):
        a_element = self.score_element()
        a_bond = self.score_bond()
        a_ring = self.score_ring()
        a_charge = self.score_charge()

        return a_element + a_charge + np.minimum(2, a_bond + 2 * a_ring)
    
    def score_mols(self, mol_q, mol_t):
        self.at_q = ZAtomType(mol_q)
        self.at_t = ZAtomType(mol_t)

        return self.score()
    
    def score_atom_types(self, at_q, at_t):
        self.at_q = at_q
        self.at_t = at_t

        return self.score()




def sdf_similarity(fp_sdf_q, fp_sdf_t):
    """
    fp_sdf_q: str, path to sdf file
    fp_sdf_t: str, path to sdf file
    """
    mol_q = Chem.SDMolSupplier(fp_sdf_q)[0]
    mol_t = Chem.SDMolSupplier(fp_sdf_t)[0]

    return ZScore(mol_q, mol_t).score()


if __name__ == "__main__":
    print(sdf_similarity(
        "/home/lccdp/Projects/clion/Uni-Dock2/unidock/unidock_engine/examples/align_Bace/CAT-13g.sdf",
        "/home/lccdp/Projects/clion/Uni-Dock2/unidock/unidock_engine/examples/align_Bace/reference.sdf"
    ))