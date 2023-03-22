# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2019 OGB Team
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file has been modified by Graphcore Ltd.

import logging
import tarfile
import math
import os
import os.path as osp
import shutil
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as AllChem
import torch
from ogb.lsc import PCQM4Mv2Dataset
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from tqdm import tqdm
from ogb.utils.url import download_url

fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# ===================== NODE START =====================

# Original features OGB-LSC (+9)
atomic_num_list = list(range(1, 119)) + ["misc"]
chiral_tag_list = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"]
possible_formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"]
possible_numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"]
possible_number_radical_e_list = [0, 1, 2, 3, 4, "misc"]
possible_hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
possible_is_aromatic_list = [False, True]
possible_is_in_ring_list = [False, True]

# Newly added simple node features (+4)
explicit_valence_list = list(range(13))
implicit_valence_list = list(range(13))
total_valence_list = list(range(26))
total_degree_list = list(range(32))

# Features from periodic table: default valence, outer electrons, van der waals radius, covalent radius (+4)
default_valence_list = list(range(-1, 5))
n_outer_electrons_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"]
rvdw_list = list(range(7))
rb0_list = list(range(9))

# Number of bonds of a particular radius around an atom (+7)
env2_list = list(range(18))
env3_list = list(range(25))
env4_list = list(range(32))
env5_list = list(range(39))
env6_list = list(range(42))
env7_list = list(range(46))
env8_list = list(range(47))

# Gasteiger charge (+1)
gasteiger_list = list(range(33))
gasteiger_list.append("misc")

# Donor/Acceptor (+2)
donor_list = [False, True]
acceptor_list = [False, True]

# Is chiral center (+1)
chiral_centers_list = ["R", "S"]
num_chiral_centers_list = list(range(3))


def simple_atom_feature(atom):
    atomic_num = safe_index(atomic_num_list, atom.GetAtomicNum())
    chiral_tag = chiral_tag_list.index(str(atom.GetChiralTag()))
    degree = safe_index(degree_list, atom.GetTotalDegree())
    possible_formal_charge = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
    possible_numH = safe_index(possible_numH_list, atom.GetTotalNumHs())
    possible_number_radical_e = safe_index(possible_number_radical_e_list, atom.GetNumRadicalElectrons())
    possible_hybridization = safe_index(possible_hybridization_list, str(atom.GetHybridization()))
    possible_is_aromatic = possible_is_aromatic_list.index(atom.GetIsAromatic())
    possible_is_in_ring = possible_is_in_ring_list.index(atom.IsInRing())
    explicit_valence = atom.GetExplicitValence()
    implicit_valence = atom.GetImplicitValence()
    total_valence = atom.GetTotalValence()
    total_degree = atom.GetTotalDegree()

    assert atomic_num in [atomic_num_list.index(l) for l in atomic_num_list]
    assert chiral_tag in [chiral_tag_list.index(l) for l in chiral_tag_list]
    assert degree in [degree_list.index(l) for l in degree_list]
    assert possible_formal_charge in [possible_formal_charge_list.index(l) for l in possible_formal_charge_list]
    assert possible_numH in [possible_numH_list.index(l) for l in possible_numH_list]
    assert possible_number_radical_e in [
        possible_number_radical_e_list.index(l) for l in possible_number_radical_e_list
    ]
    assert possible_hybridization in [possible_hybridization_list.index(l) for l in possible_hybridization_list]
    assert possible_is_aromatic in [possible_is_aromatic_list.index(l) for l in possible_is_aromatic_list]
    assert possible_is_in_ring in [possible_is_in_ring_list.index(l) for l in possible_is_in_ring_list]
    assert explicit_valence in explicit_valence_list
    assert implicit_valence in implicit_valence_list
    assert total_valence in total_valence_list
    assert total_degree in total_degree_list

    sparse_features = [
        atomic_num,
        chiral_tag,
        degree,
        possible_formal_charge,
        possible_numH,
        possible_number_radical_e,
        possible_hybridization,
        possible_is_aromatic,
        possible_is_in_ring,
        explicit_valence,
        implicit_valence,
        total_valence,
        total_degree,
    ]
    return sparse_features


def easy_bin(x, bin):
    x = float(x)
    cnt = 0
    if math.isinf(x):
        return 120
    if math.isnan(x):
        return 121

    while True:
        if cnt == len(bin):
            return cnt
        if x > bin[cnt]:
            cnt += 1
        else:
            return cnt


def peri_features(atom, peri):
    rvdw = peri.GetRvdw(atom.GetAtomicNum())
    default_valence = peri.GetDefaultValence(atom.GetAtomicNum())
    default_valence_new = safe_index(default_valence_list, default_valence)
    n_outer_elecs = peri.GetNOuterElecs(atom.GetAtomicNum())
    n_outer_elecs_new = safe_index(n_outer_electrons_list, n_outer_elecs)
    rb0 = peri.GetRb0(atom.GetAtomicNum())
    sparse_features = [
        default_valence_new,
        n_outer_elecs_new,
        easy_bin(rvdw, [1.2, 1.5, 1.55, 1.6, 1.7, 1.8, 2.4]),
        easy_bin(rb0, [0.33, 0.611, 0.66, 0.7, 0.77, 0.997, 1.04, 1.54]),
    ]
    assert default_valence_new in [default_valence_list.index(l) for l in default_valence_list]
    assert n_outer_elecs_new in [n_outer_electrons_list.index(l) for l in n_outer_electrons_list]
    assert easy_bin(rvdw, [1.2, 1.5, 1.55, 1.6, 1.7, 1.8, 2.4]) in rvdw_list
    assert easy_bin(rb0, [0.33, 0.611, 0.66, 0.7, 0.77, 0.997, 1.04, 1.54]) in rb0_list

    return sparse_features


def envatom_feature(mol, radius, atom_idx):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx, useHs=True)
    submol = Chem.PathToSubmol(mol, env, atomMap={})
    return submol.GetNumAtoms()


def envatom_features(mol, atom):
    all_envs = [env2_list, env3_list, env4_list, env5_list, env6_list, env7_list, env8_list]
    all_envs_vector = []
    for r in range(2, 9):
        all_envs_vector = envatom_feature(mol, r, atom.GetIdx())
        assert all_envs_vector in all_envs[r - 2]
    return [envatom_feature(mol, r, atom.GetIdx()) for r in range(2, 9)]


def atom_to_feature_vector(atom, peri, mol):
    sparse_features = []

    sparse_features.extend(simple_atom_feature(atom))

    sparse_features.extend(peri_features(atom, peri))

    sparse_features.extend(envatom_features(mol, atom))

    gasteiger = easy_bin(
        atom.GetProp("_GasteigerCharge"),
        [
            -0.87431233,
            -0.47758285,
            -0.38806704,
            -0.32606976,
            -0.28913129,
            -0.25853269,
            -0.24494531,
            -0.20136365,
            -0.12197541,
            -0.08234462,
            -0.06248558,
            -0.06079668,
            -0.05704827,
            -0.05296379,
            -0.04884997,
            -0.04390136,
            -0.03881107,
            -0.03328515,
            -0.02582824,
            -0.01916618,
            -0.01005982,
            0.0013529,
            0.01490858,
            0.0276433,
            0.04070013,
            0.05610381,
            0.07337645,
            0.08998278,
            0.11564625,
            0.14390777,
            0.18754518,
            0.27317209,
            1.0,
        ],
    )
    gasteiger = safe_index(gasteiger_list, gasteiger)
    sparse_features.append(gasteiger)
    assert gasteiger in [gasteiger_list.index(l) for l in gasteiger_list]
    return sparse_features


# Donor/Acceptor (+2)
def donor_acceptor_feature(x_num, mol):
    chem_feature_factory_feats = chem_feature_factory.GetFeaturesForMol(mol)
    features = np.zeros([x_num, 2], dtype=np.int64)
    for i in range(len(chem_feature_factory_feats)):
        if chem_feature_factory_feats[i].GetFamily() == "Donor":
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 0] = 1
        elif chem_feature_factory_feats[i].GetFamily() == "Acceptor":
            node_list = chem_feature_factory_feats[i].GetAtomIds()
            for j in node_list:
                features[j, 1] = 1
    return features


# Is chiral center (+1)
def chiral_centers_feature(x_num, mol):
    features = np.zeros([x_num, 1], dtype=np.int64)
    t = Chem.FindMolChiralCenters(mol)
    for i in t:
        idx, type = i
        features[idx] = chiral_centers_list.index(type) + 1  # 0 for not center
    return features


# ===================== NODE END =====================


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_feature_dims():
    return list(
        map(
            len,
            [
                atomic_num_list,
                chiral_tag_list,
                degree_list,
                possible_formal_charge_list,
                possible_numH_list,
                possible_number_radical_e_list,
                possible_hybridization_list,
                possible_is_aromatic_list,
                possible_is_in_ring_list,
                explicit_valence_list,
                implicit_valence_list,
                total_valence_list,
                total_degree_list,
                default_valence_list,
                n_outer_electrons_list,
                rvdw_list,
                rb0_list,
                env2_list,
                env3_list,
                env4_list,
                env5_list,
                env6_list,
                env7_list,
                env8_list,
                gasteiger_list,
                donor_list,
                acceptor_list,
                num_chiral_centers_list,
            ],
        )
    )


# ===================== BOND START =====================
possible_bond_type_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
possible_bond_stereo_list = [
    "STEREONONE",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOANY",
]
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))


def bond_to_feature_vector(bond):
    # 0
    bond_type = str(bond.GetBondType())
    assert bond_type in possible_bond_type_list
    bond_type = possible_bond_type_list.index(bond_type)

    bond_stereo = str(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list
    bond_stereo = possible_bond_stereo_list.index(bond_stereo)

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature


# ===================== BOND END =====================


def get_bond_feature_dims():
    return list(
        map(
            len,
            [
                possible_bond_type_list,
                possible_bond_stereo_list,
                possible_is_conjugated_list,
                possible_is_in_ring_list,
                possible_bond_dir_list,
            ],
        )
    )


def smiles2graph_large(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    AllChem.ComputeGasteigerCharges(mol)
    peri = Chem.rdchem.GetPeriodicTable()

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom, peri, mol))
    x = np.array(atom_features_list, dtype=np.int64)
    x = np.concatenate([x, donor_acceptor_feature(x.shape[0], mol)], axis=1)
    x = np.concatenate([x, chiral_centers_feature(x.shape[0], mol)], axis=1)
    for i in x:
        assert i[-1] in num_chiral_centers_list
        assert i[-2] in donor_list
        assert i[-3] in acceptor_list

    # bonds
    num_bond_features = 5
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COUP format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def get_conformers(mol_list_ogb, idx):
    conformer_positions_all_ogb = []
    for mol in range(len(idx)):
        molecule_ogb = mol_list_ogb[mol]
        conformer_positions = []
        for i, atom in enumerate(molecule_ogb.GetAtoms()):
            positions_ogb = molecule_ogb.GetConformer().GetAtomPosition(i)
            conformer_positions.append([positions_ogb.x, positions_ogb.y, positions_ogb.z])
        conformer_positions_all_ogb.append(np.array(conformer_positions, dtype=np.float32))
    return conformer_positions_all_ogb


class CustomPCQM4Mv2Dataset(PCQM4Mv2Dataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph_large,
        only_smiles=False,
        use_conformers=False,
        use_extended_features=False,
        num_processes=240,
        trim_chemical_features=False,
        use_periods_and_groups=False,
        do_not_use_atomic_number=False,
        chemical_node_features=["atomic_num"],
        chemical_edge_features=["possible_bond_type"],
        split=None,
        ensemble=False,
        load_ensemble_cache=True,
        split_mode="original",
        split_num=0,
        split_path="./pcqm4mv2-cross_val_splits/",
    ):

        self.smiles2graph = smiles2graph
        self.use_extended_features = use_extended_features
        self.use_conformers = use_conformers
        self.original_root = root
        self.only_smiles = only_smiles
        self.version = 1
        self.num_processes = num_processes
        # This flag is not actually used here
        self.trim_chemical_features = trim_chemical_features
        # This flag is not actually used here
        self.use_periods_and_groups = use_periods_and_groups
        # This flag is not actually used here
        self.do_not_use_atomic_number = do_not_use_atomic_number
        # This flag is not actually used here
        self.chemical_node_features = chemical_node_features
        # This flag is not actually used here
        self.chemical_edge_features = chemical_edge_features
        self.split = split
        self.ensemble = ensemble
        self.load_ensemble_cache = load_ensemble_cache

        self.split_mode = split_mode
        self.split_num = split_num
        self.split_path = split_path
        self.name = "pcqm4mv2"
        logging.info(f"Extended features: {use_extended_features}")
        logging.info(f"Conformers: {use_conformers}")
        if use_conformers:
            self.name += "_conformers"
        if use_extended_features:
            self.name += "_28features"
            self.smiles2graph = smiles2graph_large

        self.folder = osp.join(root, self.name)
        if not self.ensemble:
            if not osp.exists(self.folder):
                os.mkdir(self.folder)

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"

        # check version and update if necessary
        if not self.ensemble:
            if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))):
                logging.info("PCQM4Mv2 dataset has been updated.")
                if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                    shutil.rmtree(self.folder)

        super(PCQM4Mv2Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        if self.only_smiles:
            self.prepare_smiles()
        else:
            if self.ensemble:
                self.prepare_ensemble_graph()
            else:
                self.prepare_graph()

    def load_smile_strings(self, with_labels=False):
        raw_dir = osp.join(self.folder, "raw")
        data_df = pd.read_csv(osp.join(raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]
        return smiles_list, homolumogap_list

    def prepare_graph(self):
        name = self.name
        processed_dir = osp.join(self.folder, "processed")
        raw_dir = osp.join(self.folder, "raw")
        pre_processed_file_path = osp.join(processed_dir, "data_processed")

        if osp.exists(pre_processed_file_path):
            logging.info(f"Processed Dataset {pre_processed_file_path} already exists")
            # if pre-processed file already exists
            loaded_dict = torch.load(pre_processed_file_path, "rb")
            self.graphs, self.labels = loaded_dict["graphs"], loaded_dict["labels"]
        else:
            # if pre-processed file does not exist
            if not osp.exists(osp.join(raw_dir, "data.csv.gz")):
                # if the raw file does not exist, then download it.
                self.download()
                Path(self.original_root).joinpath("pcqm4m-v2").rename(Path(self.original_root).joinpath(name))

            data_df = pd.read_csv(osp.join(raw_dir, "data.csv.gz"))
            smiles_list = data_df["smiles"]
            homolumogap_list = data_df["homolumogap"]

            if self.split_mode == "incl_half_valid":
                split_file = self.split_path + "incl_half_valid/split_dict_" + str(self.split_num) + ".pt"
                train_idx = torch.load(split_file)["train"]
            elif self.split_mode == "47_kfold":
                split_file = self.split_path + "47_kfold/split_dict_" + str(self.split_num) + ".pt"
                train_idx = torch.load(split_file)["train"]
            elif self.split_mode == "train_plus_valid":
                split_file = self.split_path + "train_plus_valid/split_dict.pt"
                self.dataset.split_dict = torch.load(split_file)
            else:
                train_idx = self.get_idx_split()["train"]

            logging.info(f"Converting SMILES strings into graphs using {self.smiles2graph}")
            self.graphs = []
            self.labels = []

            if self.use_conformers:
                canonical_smiles = smiles_list
                if osp.exists(osp.join(self.folder, "pcqm4m-v2-train.sdf")):
                    logging.info("3D molecule file pcqm4m-v2-train.sdf already exists")
                else:
                    logging.info("3D molecule file pcqm4m-v2-train.sdf does not exist and will be downloaded")
                    download_url("http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz", self.folder)
                    molecule_file = tarfile.open(osp.join(self.folder, "pcqm4m-v2-train.sdf.tar.gz"))
                    molecule_file.extractall(self.folder)
                    molecule_file.close()
                suppl = Chem.SDMolSupplier(osp.join(self.folder, "pcqm4m-v2-train.sdf"), removeHs=True)
                mol_list_ogb = []

                for idx, mol in enumerate(suppl):
                    Chem.MolToSmiles(mol)
                    mol_list_ogb.append(mol)
                ogb_conformers = get_conformers(mol_list_ogb, train_idx)

                with Pool(processes=self.num_processes) as pool:
                    iter = pool.imap(self.smiles2graph, smiles_list)
                    for idx, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                        smiles = smiles_list[idx]
                        homolumogap = homolumogap_list[idx]
                        if idx < len(train_idx):
                            ogb_conformer = ogb_conformers[idx]
                        else:
                            ogb_conformer = np.array(np.full([len(graph["node_feat"]), 3], np.nan), dtype=float)

                        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                        assert len(graph["node_feat"]) == graph["num_nodes"]

                        graph["ogb_conformer"] = ogb_conformer
                        self.graphs.append(graph)
                        self.labels.append(homolumogap)
            else:
                with Pool(processes=self.num_processes) as pool:
                    iter = pool.imap(self.smiles2graph, smiles_list)
                    for idx, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                        smiles = smiles_list[idx]
                        homolumogap = homolumogap_list[idx]

                        assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                        assert len(graph["node_feat"]) == graph["num_nodes"]

                        self.graphs.append(graph)
                        self.labels.append(homolumogap)

            self.labels = np.array(self.labels)
            logging.info(f"Labels: {len(self.labels)}")

            # double-check prediction target
            if self.split_mode == "incl_half_valid":
                split_file = self.split_path + "incl_half_valid/split_dict_" + str(self.split_num) + ".pt"
                split_dict = torch.load(split_file)
            elif self.split_mode == "47_kfold":
                split_file = self.split_path + "47_kfold/split_dict_" + str(self.split_num) + ".pt"
                split_dict = torch.load(split_file)
            elif self.split_mode == "train_plus_valid":
                split_file = self.split_path + "train_plus_valid/split_dict.pt"
                self.dataset.split_dict = torch.load(split_file)
            else:
                split_dict = self.get_idx_split()

            assert all([not np.isnan(self.labels[i]) for i in split_dict["train"]])
            assert all([not np.isnan(self.labels[i]) for i in split_dict["valid"]])
            assert all([np.isnan(self.labels[i]) for i in split_dict["test-dev"]])
            assert all([np.isnan(self.labels[i]) for i in split_dict["test-challenge"]])
            logging.info("Saving...")
            torch.save({"graphs": self.graphs, "labels": self.labels}, pre_processed_file_path, pickle_protocol=4)

    def prepare_ensemble_graph(self):
        # Workaround for the issue that the folder is created with name pcqm4mv2_28features
        # but the dataset downloaded has name pcqm4m-v2
        name = "pcqm4mv2"
        self.folder = osp.join(self.original_root, name)
        processed_dir = osp.join(self.folder, "processed")
        raw_dir = osp.join(self.folder, "raw")
        data_processed_file_name = "data_processed"
        # The processed graphs from SMILE string will be different
        # if use_extended_features is true.
        if self.use_extended_features:
            data_processed_file_name += "_28features"
        data_processed_file_name += f"_{self.split_mode}"
        if self.split is not None:
            data_processed_file_name += "_" + "_".join(self.split)
        pre_processed_file_path = osp.join(processed_dir, data_processed_file_name)
        logging.info(f"processed_dir, {processed_dir}")
        logging.info(f"pre_processed_file_path, {pre_processed_file_path}")

        if osp.exists(pre_processed_file_path) and self.load_ensemble_cache:
            logging.info(f"Processed Dataset {pre_processed_file_path} already exists")
            # if pre-processed file already exists
            loaded_dict = torch.load(pre_processed_file_path, "rb")
            self.graphs, self.labels = loaded_dict["graphs"], loaded_dict["labels"]

        else:
            # if pre-processed file does not exist
            if not osp.exists(osp.join(raw_dir, "data.csv.gz")):
                # If the raw file does not exist, download it.
                self.download()
                # Rename folder from "pcqm4m-v2" to name = 'pcqm4mv2'
                # to match the dataset_name in the cache loading section
                Path(self.original_root).joinpath("pcqm4m-v2").rename(Path(self.original_root).joinpath(name))

            data_df = pd.read_csv(osp.join(raw_dir, "data.csv.gz"))
            self.split_indices = []

            # split_dict would change based on the split_mode used
            if self.split_mode == "incl_half_valid":
                split_file = self.split_path + "incl_half_valid/split_dict_" + str(self.split_num) + ".pt"
                split_dict = torch.load(split_file)
            elif self.split_mode == "47_kfold":
                raise NotImplementedError(f"The split mode of {self.split_mode} is not implemented with ensembling.")
            elif self.split_mode == "train_plus_valid":
                split_file = self.split_path + "train_plus_valid/split_dict.pt"
                split_dict = torch.load(split_file)
                # If train_plus_valid is used for training, it is not necessary to do validation, therefore we pop
                # the indices out from split_dict with the ensembler.
                split_dict.pop("valid")
            else:
                # split mode original
                split_dict = self.get_idx_split()

            # Get the indces for valid splits
            for split in self.split:
                self.split_indices += split_dict[split].tolist()

            smiles_sub_list = data_df["smiles"][self.split_indices]
            homolumogap_list = data_df["homolumogap"]
            homolumogap_sub_list = homolumogap_list[self.split_indices]

            # Initialize sub_graphs and sub_labels as dictionary to keep track of the original indices.
            self.sub_graphs = {}
            self.sub_labels = {}
            full_dataset_length = len(homolumogap_list)
            self.graphs = [
                {"edge_index": np.nan, "edge_feat": np.nan, "node_feat": np.nan, "num_nodes": np.nan}
                for i in range(full_dataset_length)
            ]
            self.labels = [np.nan for i in range(full_dataset_length)]

            logging.info("Converting SMILES strings into graphs for sub-sets...")
            # Process the SMILE string for sub-sets in parallel
            # But the final time difference by using 1-240 cores is minor (10s - 60s).
            num_cores = 240
            with Pool(processes=num_cores) as pool:
                iter = pool.imap(self.smiles2graph, smiles_sub_list)
                for idx, graph in tqdm(enumerate(iter), total=len(homolumogap_sub_list)):
                    smiles_index = smiles_sub_list.keys()[idx]
                    homolumo_index = homolumogap_sub_list.keys()[idx]
                    assert (
                        smiles_index == homolumo_index
                    ), "The index for smiles string and index for homolumo gap do not match. Please check your dataset."

                    homolumogap = homolumogap_sub_list[homolumo_index]

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    if self.use_conformers:
                        # Just place-holder in case the model pre-processing steps complain about it
                        graph["ogb_conformer"] = np.zeros([len(graph["node_feat"]), 3], dtype=float)

                    self.graphs[smiles_index] = graph
                    self.labels[homolumo_index] = homolumogap

            # Convert to numpy array
            self.labels = np.array(self.labels)

            logging.info(f"Labels: {len(self.labels)}")
            for split in self.split:
                if split == "valid":
                    assert all([not np.isnan(self.labels[i]) for i in split_dict["valid"]])
                # The labels in 'train' will be nan as 'train' split will not be included in the splits for ensembling
                if split in ("train", "test-dev", "test-challenge"):
                    assert all([np.isnan(self.labels[i]) for i in split_dict["test-challenge"]])
            logging.info(f"Saving data for split/splits {self.split}.")
            torch.save({"graphs": self.graphs, "labels": self.labels}, pre_processed_file_path, pickle_protocol=4)
