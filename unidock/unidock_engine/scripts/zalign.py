from rdkit import Chem
from rdkit.Chem import AllChem


class ZAtomType:
    """获取特征"""
    
    mol = None

    def read_sdf(file_path):
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
            atom_type = 7

            if i not in dict_atom_ringsize:
                atom_type = 0
            
            elif len(dict_atom_ringsize[i]) == 1:
                if ring_info.IsAtomInRingOfSize(i, 3):
                    atom_type = 1
                elif ring_info.IsAtomInRingOfSize(i, 4):
                    atom_type = 2
                elif ring_info.IsAtomInRingOfSize(i, 5):
                    atom_type = 3
                elif ring_info.IsAtomInRingOfSize(i, 6):
                    atom_type = 4
            
            elif len(dict_atom_ringsize[i]) == 2:
                if set(dict_atom_ringsize[i]) == {5, 6}:
                    atom_type = 5
                elif set(dict_atom_ringsize[i]) == {6}:
                    atom_type = 6

        
            atom_ring_types.append(atom_type)

        return atom_ring_types


    def get_atom_charge_types(self, ff=False):
        # 补氢
        self.mol = Chem.AddHs(self.mol)  
        
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

        return atom_charges


    def get_atom_bond_number(self):
        atom_bond_number = [len(atom.GetBonds()) for atom in self.mol.GetAtoms()]
        return atom_bond_number


    def get_atom_element_types(self):
        atom_element_types = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        return atom_element_types
    

