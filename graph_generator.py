import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
import pickle
import matplotlib.pyplot as plt
from molecule.molecule import Molecule, Atom, Bond
import yaml

def mol_to_nx(mol):
    """将 RDKit 分子转换为 NetworkX 图"""
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=bond.GetBondTypeAsDouble())
    return G

def nx_to_mol(G):
    """将 NetworkX 图转换为 RDKit 分子"""
    mol = Chem.RWMol()
    idx_map = {}
    for node, data in G.nodes(data=True):
        atom = Chem.Atom(data['symbol'])
        idx = mol.AddAtom(atom)
        idx_map[node] = idx
    for start, end, data in G.edges(data=True):
        mol.AddBond(idx_map[start], idx_map[end], Chem.BondType(data['order']))
    return mol

def nx_to_rmg(graph):
    rmg_str = "multiplicity 1\n\n"
    flag_idx = 1
    for node, data in graph.nodes(data=True):
        if data['flag']:
            flag = f'*{flag_idx}'
            flag_idx += 1
        else:
            flag = '  '
        rmg_str += f"    {node} {flag} {data['symbol']} u{int(data['unpaired_electrons'])} p{int(data['lone_pairs'])} c{int(data['charge'])} "
        for neighbor in graph.neighbors(node):
            bond = graph[node][neighbor]['bond']
            rmg_str += f"{{{neighbor},{bond}}} "
        rmg_str += "\n"
    return rmg_str
    

def draw_molecule(G):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, labels={node: data.get('symbol', '') for node, data in G.nodes(data=True)}, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'bond')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

def break_bond(G, atom1_idx, atom2_idx):
    """断裂 NetworkX 图中的化学键"""
    for node, data in G.nodes(data=True):
        if 'flag' in data:
            del G.nodes[node]['flag']
    G_copy = G.copy()
    # 在atom1_idx对应的原子和atom2_idx对应的原子上添加标记
    G.nodes[atom1_idx]['flag'] = True
    G.nodes[atom2_idx]['flag'] = True
    G_copy.nodes[atom1_idx]['flag'] = True
    G_copy.nodes[atom2_idx]['flag'] = True
    G_copy.remove_edge(atom1_idx, atom2_idx)
    components = list(nx.connected_components(G_copy))
    subgraphs = [G_copy.subgraph(c).copy() for c in components]
    return subgraphs, G

def add_bond(G1, G2, atom1_idx, atom2_idx):
    """在两个 NetworkX 图之间添加化学键"""
    for node, data in G1.nodes(data=True):
        if 'flag' in data:
            del G1.nodes[node]['flag']
    for node, data in G2.nodes(data=True):
        if 'flag' in data:
            del G2.nodes[node]['flag']
    
    G1_copy = G1.copy()
    G2_copy = G2.copy()

    # 获取 G2 中节点的偏移量
    offset = max(G1_copy.nodes) + 1
    
    # 更新 G2 中节点的索引，保留原先的symbol属性
    mapping = {node: node + offset for node in G2_copy.nodes}
    G2_copy = nx.relabel_nodes(G2_copy, mapping)

    G = nx.compose(G1_copy, G2_copy)
    
    # 添加一个新的化学键
    G.add_edge(atom1_idx, atom2_idx + offset, order=1.0)

    # 添加标记
    G1.nodes[atom1_idx]['flag'] = True
    G2.nodes[atom2_idx]['flag'] = True
    G.nodes[atom1_idx]['flag'] = True
    G.nodes[atom2_idx + offset]['flag'] = True
    
    return G, [G1, G2]

def is_isomorphic(G1, G2):
    """判断两个分子是否相同"""
    return nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1['symbol'] == n2['symbol'])

def get_molecule(formula):
    """从分子式生成 NetworkX 图"""
    if formula == 'H2O':
        smiles = 'O'
    elif formula == 'CO':
        G = nx.Graph()
        G.add_node(0, symbol='C')
        G.add_node(1, symbol='O')
        G.add_edge(0, 1, order=2.0)
        return G
    elif formula == 'H2':
        smiles = '[H][H]'
    elif formula == 'CO2':
        smiles = 'O=C=O'
    elif formula == 'CH4':
        smiles = 'C'
    elif formula == 'C2H6':
        smiles = 'CC'
    elif formula == 'CH3OH':
        smiles = 'CO'
    elif formula == 'CH3CH2OH':
        smiles = 'CCO'
    else:
        smiles = None
            
    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    G = mol_to_nx(mol)

    return G

def oxygen_constraint(G):
    """确保一个 O 原子最多只能与一个 C 原子键合，并且不包括含有 O-O 键的中间体"""
    for node, data in G.nodes(data=True):
        if data['symbol'] == 'O':
            c_count = 0
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['symbol'] == 'C':
                    c_count += 1
                if G.nodes[neighbor]['symbol'] == 'O':
                    return True
            if c_count > 1:
                return True
    return False

def carbon_oxygen_constraint(G):
    """确保所有中间体的碳原子数不超过 2 个，中间体中 C 和 O 原子的总和不超过 3"""
    carbon_count = sum(1 for _, data in G.nodes(data=True) if data['symbol'] == 'C')
    oxygen_count = sum(1 for _, data in G.nodes(data=True) if data['symbol'] == 'O')
    if carbon_count > 2 or (carbon_count + oxygen_count) > 3:
        return True
    return False

def carbon_oh_constraint(G):
    """确保一个碳原子最多只能连接到一个 OH 基团"""
    for node, data in G.nodes(data=True):
        if data['symbol'] == 'C':
            oh_count = 0
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['symbol'] == 'O':
                    for oh_neighbor in G.neighbors(neighbor):
                        if G.nodes[oh_neighbor]['symbol'] == 'H':
                            oh_count += 1
            if oh_count > 1:
                return True
    return False

def get_name(G):
    # 获取能表示分子结构的化学式，例如：‘CH3CH2OH’
    formula = ''
    for node, data in G.nodes(data=True):
        if data['symbol'] == 'C' or data['symbol'] == 'O':
            formula += data['symbol']
            H_count = sum(1 for n in G.neighbors(node) if G.nodes[n]['symbol'] == 'H')
            if H_count == 0:
                pass
            elif H_count == 1:
                formula += 'H'
            else:
                formula += 'H' + str(H_count)
    if formula == '':
        for node, data in G.nodes(data=True):
            formula += data['symbol']
    return formula

full_connect_num = {
    'H': 1,
    'C': 4,
    'O': 2,
}

elec_num = {
    'H': 1,
    'C': 4,
    'O': 6,
}

class ReactionNetwork:
    def __init__(self, species, target, constrains):
        self.network = nx.Graph()
        self.species = species
        self.target = target
        self.products = []
        self.reactions = []
        self.constrains = constrains

        for s in self.species:
            self.network.add_node(s, name=get_name(s))
    
    # 化合反应
    def bonding(self, r1, r2):
        if self.is_target(r1) or self.is_target(r2):
            return

        # 从 r1 和 r2 中找到可以连接的原子
        r1_atoms = [node for node, data in r1.nodes(data=True) if len(list(r1.neighbors(node))) < full_connect_num[data['symbol']]]
        r2_atoms = [node for node, data in r2.nodes(data=True) if len(list(r2.neighbors(node))) < full_connect_num[data['symbol']]]

        for idx1 in r1_atoms:
            for idx2 in r2_atoms:
                p, [r1, r2] = add_bond(r1, r2, idx1, idx2)
                # 如果生成的分子是不重复的，添加到结果中
                if self.check(p):
                    self.products.append(p)
                    self.network.add_node(p, name=get_name(p))  # 添加产物到网络中

                # 获得节点数目，节点少的为r1
                # if len(r1) < len(r2):
                #     rec = ('2->1', (r1, r2), p)
                # else:
                #     rec = ('2->1', (r2, r1), p)
                rec = ('2->1', (r1, r2), p)

                if self.check_rec(rec):
                    self.reactions.append(rec)
                    for node, data in self.network.nodes(data=True):
                        if is_isomorphic(p, node):
                            self.network.add_edge(r1, node, reaction=rec)
                            self.network.add_edge(r2, node, reaction=rec)
                    
    
    def bonding_self(self, r):
        # TODO: 实现自身连接的反应
        pass
        
    # 解离反应
    def break_bond(self, r):
        r_atoms = [node for node, data in r.nodes(data=True) if len(list(r.neighbors(node))) > 0]
        for idx1 in r_atoms:
            for idx2 in r.neighbors(idx1):
                ps, r = break_bond(r, idx1, idx2)
                if len(ps) == 2:
                    p1, p2 = ps
                    f1 = self.check(p1)
                    f2 = self.check(p2) and not is_isomorphic(p1, p2)
                    if f1:
                        self.products.append(p1)
                        self.network.add_node(p1, name=get_name(p1))  # 添加产物到网络中
                        
                    if f2:
                        self.products.append(p2)
                        self.network.add_node(p2, name=get_name(p2))  # 添加产物到网络中
                        
                    # if len(p1) < len(p2):
                    #     rec = ('1->2', r, (p1, p2))
                    # else:
                    #     rec = ('1->2', r, (p2, p1))

                    rec = ('1->2', r, (p1, p2))
                    
                    if self.check_rec(rec):
                        self.reactions.append(rec)
                        for node, data in self.network.nodes(data=True):
                            if is_isomorphic(p1, node):
                                self.network.add_edge(r, node, reaction=rec)
                            if is_isomorphic(p2, node):
                                self.network.add_edge(r, node, reaction=rec)

                elif len(ps) == 1:
                    p = ps[0]
                    if self.check(p):
                        self.products.append(p)
                        
                    rec = ('1->1', r, p)
                
                    if self.check_rec(rec):
                        self.reactions.append(rec)
                        for node, data in self.network.nodes(data=True):
                            if is_isomorphic(p, node):
                                self.network.add_edge(r, node, reaction=rec)

    def is_repeat(self, G):
        for s in self.species:
            if is_isomorphic(G, s):
                return True
        for p in self.products:
            if is_isomorphic(G, p):
                return True
        return False

    def is_repeat_rec(self, rec):
        for r in self.reactions:
            if rec[0] == r[0]:
                if r[0] == '2->1':
                    # 排序后比较
                    sorted_rec1 = sorted(list(rec[1]), key=lambda x: len(x.nodes))
                    sorted_r1 = sorted(list(r[1]), key=lambda x: len(x.nodes))
                    if is_isomorphic(sorted_rec1[0], sorted_r1[0]) and is_isomorphic(sorted_rec1[1], sorted_r1[1]) and is_isomorphic(rec[2], r[2]):
                        return True
                elif r[0] == '1->2':
                    sorted_rec2 = sorted(list(rec[2]), key=lambda x: len(x.nodes))
                    sorted_r2 = sorted(list(r[2]), key=lambda x: len(x.nodes))
                    if is_isomorphic(rec[1], r[1]) and is_isomorphic(sorted_rec2[0], sorted_r2[0]) and is_isomorphic(sorted_rec2[1], sorted_r2[1]):
                        return True
                elif r[0] == '1->1':
                    if is_isomorphic(rec[1], r[1]) and is_isomorphic(rec[2], r[2]):
                        return True
        return False

    def generate(self):
        for i in range(len(self.species)):
            for j in range(i + 1, len(self.species)):
                self.bonding(self.species[i], self.species[j])
        for i in range(len(self.species)):
            self.break_bond(self.species[i])
        self.species += self.products
        self.products = []

    def run(self):
        species_num = len(self.species)
        while True:
            self.generate()
            if species_num == len(self.species):
                break
            species_num = len(self.species)

    def is_target(self, s):
        for t in self.target:
            if is_isomorphic(s, t):
                return True
        return False
    
    def constrain(self, p):
        for cons in self.constrains:
            if cons(p):
                return False
        return True
    
    def check(self, p):
        if self.is_repeat(p):
            return False
        if not self.constrain(p):
            return False
        return True

    def check_rec(self, rec):
        if self.is_repeat_rec(rec):
            return False
        if rec[0] == '2->1':
            if not self.constrain(rec[2]):
                return False
        elif rec[0] == '1->2':
            if  not self.constrain(rec[2][0]) or not self.constrain(rec[2][1]):
                return False
        elif rec[0] == '1->1':
            if not self.constrain(rec[2]):
                return False
        return True

    def draw_network(self):
        plt.figure(figsize=(24, 24))
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, with_labels=True, labels={node: data.get('name', '') for node, data in self.network.nodes(data=True)}, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
        # labels = nx.get_edge_attributes(self.network, 'reaction')
        # nx.draw_networkx_edge_labels(self.network, pos, edge_labels=labels)
        plt.show()

    def save(self):
        # 保存整个数据结构到同一个文件
        with open('network.pkl', 'wb') as f:
            pickle.dump(self, f)
    
    def load(self):
        # 从文件中加载整个数据结构
        with open('network.pkl', 'rb') as f:
            return pickle.load(f)

    def reduce_rec(self):
        reactions = []
        for rec in self.reactions:
            if rec[0] == '2->1':
                continue
            reactions.append(rec)
        self.rd_reactions = reactions

    def standardized_mol(self, G):
        for node, data in G.nodes(data=True):
            if data['symbol'] == 'O' and len(list(G.neighbors(node))) < 2:
                idx = node
                for neighbor in G.neighbors(idx):
                    if G.nodes[neighbor]['symbol'] == 'C' and len(list(G.neighbors(node))) < 4:
                        G.add_edge(idx, neighbor, order=2.0)

        for node, data in G.nodes(data=True):
            if data['symbol'] == 'X':
                idx = node
                for neighbor in G.neighbors(idx):
                    bond_num = 0
                    for n in G.neighbors(neighbor):
                        bond_num += G[neighbor][n]['order']
                    if bond_num < full_connect_num[G.nodes[neighbor]['symbol']]:
                        d = full_connect_num[G.nodes[neighbor]['symbol']] - bond_num + 1
                        G.add_edge(idx, neighbor, order=d)

        
        G_std = nx.Graph()
        for node, data in G.nodes(data=True):
            symbol = data['symbol']
            if symbol == 'X':
                G_std.add_node(node, symbol=symbol, lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False, flag=data['flag'])
                continue
            # 获得与该原子相连的边
            bonds = []
            bond_num = 0
            for neighbor in G.neighbors(node):
                bonds.append((neighbor, G[node][neighbor]['order']))
                bond_num += G[node][neighbor]['order']
            # 计算孤对数
            elec = elec_num[symbol] - bond_num
            lone_pairs = elec // 2
            unpaired_electrons = elec % 2
            charge = 0
            unfull_flag = full_connect_num[symbol] - bond_num > 0
            if 'flag' in data:
                flag = True
            else:
                flag = False
            G_std.add_node(node, symbol=symbol, lone_pairs=lone_pairs, 
                           unpaired_electrons=unpaired_electrons, charge=charge, unfull=unfull_flag, flag=flag)
        
        for node1, node2, data in G.edges(data=True):
            if data['order'] == 1.0:
                bond_type = 'S'
            elif data['order'] == 2.0:
                bond_type = 'D'
            elif data['order'] == 3.0:
                bond_type = 'T'
            elif data['order'] == 4.0:
                bond_type = 'Q'
            G_std.add_edge(node1, node2, bond=bond_type, order=data['order'])
        
        # # 添加吸附位点
        # G_tmp = G_std.copy()
        # for node, data in G_tmp.nodes(data=True):
        #     if data['unfull']:
        #         G_std.add_node(max(G_std.nodes) + 1, symbol='X', lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False)
        #         G_std.add_edge(node, max(G_std.nodes), bond='S')
        
        return G_std
            
    def standardized_rec(self, rec):
        # 将1->2字符串转换为列表[1, 2]
        # rec_type = [int(i) for i in rec[0].split('->')]
        # 将反应中的分子转换为标准格式
        reactants = rec[1]
        products = rec[2]
        
        # 将反应物的原子序号和产物的原子序号对应起来
        if isinstance(reactants, tuple):
            r = nx.compose(reactants[0], reactants[1])
        else:
            r = reactants
        if isinstance(products, tuple):
            p = nx.compose(products[0], products[1])
        else:
            p = products

        r = self.standardized_mol(r)
        p = self.standardized_mol(p)
        
        # 添加吸附位点
        # 根据idx遍历所有的节点，如果节点是未饱和的，添加一个吸附位点
        
        assert len(r.nodes) == len(p.nodes)

        r_tmp = r.copy()
        p_tmp = p.copy()

        for idx, ((node1, data1), (node2, data2)) in enumerate(zip(r_tmp.nodes(data=True), p_tmp.nodes(data=True))):
            if data1['unfull'] or data2['unfull']:
                if not (data1['unfull'] == data2['unfull']):
                    r.add_node(max(r.nodes) + 1, symbol='X', lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False, flag=True) 
                    p.add_node(max(p.nodes) + 1, symbol='X', lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False, flag=True)
                else:
                    r.add_node(max(r.nodes) + 1, symbol='X', lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False, flag=False) 
                    p.add_node(max(p.nodes) + 1, symbol='X', lone_pairs=0, unpaired_electrons=0, charge=0, unfull=False, flag=False)
                
                if data1['unfull']:
                    r.add_edge(node1, max(r.nodes), bond='S', order=1.0)

                if data2['unfull']:
                    p.add_edge(node2, max(p.nodes), bond='S', order=1.0)

        r = self.standardized_mol(r)
        p = self.standardized_mol(p)

        return r, p
    
    def get_rec_rmg(self, rec, idx):
        r, p = self.standardized_rec(rec)
        reaction = f"{get_name(r)} -> {get_name(p)}"
        r = networkx_to_rmg(r)
        p = networkx_to_rmg(p)

        
        
        # 使用yaml格式保存
        msg = {
            'index': idx,
            'product': p,
            'reactant': r,
            'reaction': reaction,
            'reaction_family': 'Surface'
        }
        return msg


def networkx_to_rmg(networkx_graph):
    """
    将NetworkX图转换为RMG Molecule对象
    """
    rmg_molecule = Molecule()
    atom_mapping = {}
    
    # 添加节点（原子）
    for node, data in networkx_graph.nodes(data=True):
        atom = Atom(element=data['symbol'])
        atom.radical_electrons = data['unpaired_electrons']
        atom.charge = data['charge']
        atom.lone_pairs = data['lone_pairs']
        rmg_molecule.add_atom(atom)
        atom_mapping[node] = atom
    
    # 添加边（键）
    for u, v, data in networkx_graph.edges(data=True):
        bond_type = data['bond']
        bond = Bond(atom_mapping[u], atom_mapping[v], order=bond_type)
        rmg_molecule.add_bond(bond)

    rmg_molecule.update_multiplicity()
    rmg_molecule.update()
    
    return rmg_molecule.to_adjacency_list()



target_formula = [
    'H2O', 'H2', 'CO2', 'CH4', 'C2H6', 'CH3OH', 'CH3CH2OH', 'CO'
]

target = [get_molecule(f) for f in target_formula]
constrains = [oxygen_constraint, carbon_oxygen_constraint, carbon_oh_constraint]

species = [get_molecule('CO'), get_molecule('H2')]

recnet = ReactionNetwork(species, target, constrains)
# recnet.run()
# recnet.save()
recnet = recnet.load()
# recnet.draw_network()

# print(len(recnet.species))
# print(len(recnet.network.nodes))
# print(len(recnet.reactions))
# print(len(recnet.network.edges))

recnet.reduce_rec()
rec = recnet.rd_reactions[0]

msg = recnet.get_rec_rmg(rec, 0)

rec2 = recnet.rd_reactions[5]
msg2 = recnet.get_rec_rmg(rec2, 1)

rec3 = recnet.rd_reactions[1]
msg3 = recnet.get_rec_rmg(rec3, 2)

with open('./test.yaml', 'w') as f:
    yaml.dump(data=[msg, msg2, msg3], stream=f)

# with open('./rxn_test.yaml', 'r') as f:
#     msg = yaml.load(f, Loader=yaml.FullLoader)
#     #print(msg)

# print(msg[0])

# 打印所有的反应
# recnet.reduce_rec()
# print(len(recnet.rd_reactions))
# for rec in recnet.rd_reactions:
#     if rec[0] == '2->1':
#         print(f"{get_name(rec[1][0])} + {get_name(rec[1][1])} -> {get_name(rec[2])}")
#     elif rec[0] == '1->2':
#         print(f"{get_name(rec[1])} -> {get_name(rec[2][0])} + {get_name(rec[2][1])}")
#     elif rec[0] == '1->1':
#         print(f"{get_name(rec[1])} -> {get_name(rec[2])}")

