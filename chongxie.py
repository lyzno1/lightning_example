
from __future__ import annotations

import os
import sys
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
import random
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything


def get_atom_distance(structure, atom_i, atom_j):  # 获取元素之间的距离
    """
        计算两个原子之间的距离
        Args:
            structure (Structure): pymatgen Structure 对象
            atom_i (int): 第一个原子的索引
            atom_j (int): 第二个原子的索引
        Returns:
            distance (float): 两个原子之间的距离
        """
    site_i = structure[atom_i]
    site_j = structure[atom_j]
    distance = site_i.distance(site_j)
    return distance

def get_atomic_number(symbol):  # 元素符号转化为原子序数
    return Element(symbol).number

class TripletStats: # 统计三元组数量
    def __init__(self, structures):
        self.structures = structures
        self.triplet_counts = self.calculate_triplet_counts()
        self.average = self.calculate_average()
        self.max_value = max(self.triplet_counts)
        self.min_value = min(self.triplet_counts)
        self.median = self.calculate_median()
        self.most_common = self.calculate_most_common()
        self.least_common = self.calculate_least_common()
        self.new_max, self.new_min = self.calculate_trimmed_extremes()

    def calculate_triplet_counts(self):
        triplet_counts = []
        for structure in self.structures:
            len_triplet = (len(structure) * (len(structure) - 1)) // 2
            triplet_counts.append(len_triplet)
        return triplet_counts

    def calculate_average(self):
        return sum(self.triplet_counts) / len(self.triplet_counts)

    def calculate_median(self):
        sorted_counts = sorted(self.triplet_counts)
        n = len(sorted_counts)
        if n % 2 == 1:
            return sorted_counts[n // 2]
        else:
            return (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2

    def calculate_most_common(self):
        from collections import Counter
        count = Counter(self.triplet_counts)
        return count.most_common(1)[0][0]

    def calculate_least_common(self):
        from collections import Counter
        count = Counter(self.triplet_counts)
        return count.most_common()[-1][0]

    def calculate_trimmed_extremes(self):
        trimmed_counts = [x for x in self.triplet_counts if x != self.max_value and x != self.min_value]
        if trimmed_counts:
            new_max = max(trimmed_counts)
            new_min = min(trimmed_counts)
            return new_max, new_min
        else:
            return None, None  # 当所有值都相同时，去除后为空列表

    def get_max_value(self):
        print("最大值:", self.max_value)
        return int(self.max_value)

    def get_min_value(self):
        print("最小值:", self.min_value)
        return int(self.min_value)

    def get_median(self):
        print("中位数:", self.median)
        return int(self.median)

    def get_average(self):
        print("平均数:", self.average)
        return int(self.average)

    def get_most_common(self):
        print("出现最多的数:", self.most_common)
        return int(self.most_common)

    def get_least_common(self):
        print("出现最少的数:", self.least_common)
        return int(self.least_common)

    def get_new_max(self):
        print("去除最大最小值之后的最大值:", self.new_max)
        return int(self.new_max)

    def get_new_min(self):
        print("去除最大最小值之后的最小值:", self.new_min)
        return int(self.new_min)

def get_triplets(structures, max_len):  # 处理成三元组
    all_tensor_data = []  # 存储所有结构的三元组数据
    for structure in structures:
        tensor_data = []
        num_atoms = len(structure)

        if num_atoms == 1:
            lattice = structure.lattice
            atom_symbol = structure[0].species_string
            atomic_number = get_atomic_number(atom_symbol)
            triplet_data = (atomic_number, atomic_number, lattice.a)
            tensor_data.append(triplet_data)
            triplet_data = (atomic_number, atomic_number, lattice.b)
            tensor_data.append(triplet_data)
            triplet_data = (atomic_number, atomic_number, lattice.c)
            tensor_data.append(triplet_data)
            all_tensor_data.append(tensor_data)
            continue

        for i in range(num_atoms):
            element_i = structure[i].species_string
            for j in range(i + 1, num_atoms):
                element_j = structure[j].species_string
                distance = get_atom_distance(structure, i, j)

                # 将原子转换为对应的原子序数
                atomic_number_i = get_atomic_number(element_i)
                atomic_number_j = get_atomic_number(element_j)
                # 存储原始的三元组数据
                triplet_data = (atomic_number_i, atomic_number_j, distance)
                tensor_data.append(triplet_data)

        # 对三元组列表按照最后一个元素（距离信息）进行升序排序
        tensor_data.sort(key=lambda x: x[2], reverse=False)

        # 截断数据到max_length长度
        if len(tensor_data) > max_len:
            tensor_data = tensor_data[:max_len]
        # 将当前结构的三元组数据添加到总列表中
        all_tensor_data.append(tensor_data)

    # 对不足最大长度的子列表进行补充
    for sublist in all_tensor_data:
        while len(sublist) < max_len:
            sublist.append((0, 0, 0.0))

    return all_tensor_data

# 距离bond expansion处理
class BondExpansionRBF(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionRBF, self).__init__()
        self.num_features = num_features
        self.gamma = gamma

    def __call__(self, bond_dist: torch.Tensor) -> torch.Tensor:
        # 生成特征中心张量
        feature_centers = torch.arange(1, self.num_features + 1, device=bond_dist.device).float()

        # 计算每个距离到各个特征中心的欧几里得距离
        distance_to_centers = torch.abs(feature_centers - bond_dist.unsqueeze(-1))

        # 使用高斯径向基函数计算每个距离对应的特征值
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2).squeeze()

        return rbf_values

class BondExpansionLearnable(nn.Module):
    def __init__(self, num_features: int = 10, gamma: float = 1.0):
        super(BondExpansionLearnable, self).__init__()
        self.num_features = num_features
        self.centers = nn.Parameter(torch.randn(num_features))
        self.gamma = gamma #nn.Parameter(torch.one(1))

    def __call__(self, bond_dist: torch.Tensor) -> torch.Tensor:
        distance_to_centers = torch.abs(self.centers - bond_dist.unsqueeze(-1))
        rbf_values = torch.exp(-self.gamma * distance_to_centers ** 2)

        return rbf_values

class StructureDataset(Dataset):
    def __init__(self, structure, target):
        self.input = structure
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target.iloc[idx]

        # 返回合金描述和目标值
        return {'input': input, 'target': target}

class Attention_structure_model(pl.LightningModule):
    def __init__(self, embedding_dim=10, hidden_size=64, output_size=1, dropout=0.2, num_features=10, gamma=1.8):
        super(Attention_structure_model, self).__init__()
        self.embedding = nn.Embedding(119, embedding_dim)
        self.bond_expansion = BondExpansionRBF(num_features=num_features, gamma=gamma)

        self.gru = nn.GRU(input_size=embedding_dim * 3, hidden_size=hidden_size, batch_first=True, num_layers=3,
                          dropout=dropout)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear_feed = nn.Linear(hidden_size, 1024)
        self.relu1 = nn.ReLU()
        self.linear_forward = nn.Linear(1024, hidden_size)

        self.self_attention1 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True, dropout=0.2)

        self.linear = nn.Linear(hidden_size, 32)
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(32, output_size)

    def forward(self, x):
        # 嵌入元素和键扩展
        atom1 = self.embedding(x[:, :, 0].to(torch.long))
        atom2 = self.embedding(x[:, :, 1].to(torch.long))
        bond = self.bond_expansion(x[:, :, 2].float())
        embedded_data = torch.cat((atom1, atom2, bond), dim=-1)

        # shape: (batch_size, seq_len, input_size)
        gru_output, _ = self.gru(embedded_data)

        attention_out, _ = self.self_attention(gru_output, gru_output, gru_output)
        attention_out = self.layer_norm(attention_out + gru_output)

        linear_out = self.linear_feed(attention_out)
        linear_out = self.relu1(linear_out)
        linear_out = self.linear_forward(linear_out)
        linear_out = self.layer_norm(attention_out + attention_out)

        attention_out, _ = self.self_attention(linear_out, linear_out, linear_out)
        attention_out = self.layer_norm(attention_out + linear_out)

        linear_out = self.linear_feed(attention_out)
        linear_out = self.relu1(linear_out)
        linear_out = self.linear_forward(linear_out)
        linear_out = self.layer_norm(linear_out + attention_out)

        attention1, _ = self.self_attention1(linear_out, linear_out, linear_out)

        output = attention1[:, -1, :]

        output = self.linear1(self.relu2(self.linear(output)))

        return output.squeeze(1)

class liu_attention_Lightning(pl.LightningModule):
    def __init__(self, hparams):
        super(liu_attention_Lightning, self).__init__()
        self.save_hyperparameters()

        self.model = Attention_structure_model(embedding_dim=hparams.embedding_dim, hidden_size=hparams.hidden_size,
                                              output_size=hparams.output_size, dropout=hparams.dropout, num_features=hparams.num_features)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.lr)
    def training_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)
        loss = F.l1_loss(predict, label)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)
        val_loss = F.l1_loss(predict, label)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, label = batch['input'], batch['target'].float()
        predict = self.model(x)

        test_loss = F.l1_loss(predict, label)
        self.log('test_mae', test_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer

def visualize_results(results_list, mb_dataset_name): # 可视化结果并保存到文件中
    for i, mae in enumerate(results_list):
        print(f"Fold {i} MAE: {mae}")
    average_mae = sum(mae_list) / len(mae_list)
    print(f"Average MAE across all folds: {average_mae}")

    # 写入结果到文件
    # with open('results.txt', 'a') as f:
    #     if f.tell() != 0:
    #         f.write('\n')
    #     for fold_num, mae in enumerate(results_list):
    #         f.write(f"Fold {fold_num}, MAE:{mae}\n")
    #     f.write(f"{mb_dataset_name}, batch_size:{batch_size}, gamma:{args.gamma}, cutoff:{args.cutoff}, Average MAE: {average_mae}\n")
    # results_list.clear()

def main(hparams):
    seed_everything(hparams.seed)
    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            # "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            # "matbench_dielectric",  # 4,764
            # "matbench_log_gvrh",  # 10,987
            # "matbench_log_kvrh",  # 10,987
            # "matbench_perovskites",  # 1w8
            # "matbench_mp_gap",   # 回归 10.6w
            # "matbench_mp_e_form",  # 回归 13w
        ]
    )

    mae_list = []
    for task in mb.tasks:
        task.load()
        dataset_name = task.dataset_name
        for fold in task.folds:

            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            max_length = TripletStats(train_inputs).get_average() * hparams.cutoff # 用于截断/补齐

            x_input = torch.tensor(get_triplets(train_inputs, max_length))  # 处理输入

            dataset = StructureDataset(x_input, train_outputs)
            train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=hparams.seed)

            train_loader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_worker,
                                      persistent_workers=True)
            val_loader = DataLoader(val_data, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_worker,
                                    persistent_workers=True)

            lightning_model = liu_attention_Lightning(hparams)

            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=hparams.patience, verbose=True,
                                                mode="min")
            checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

            trainer = pl.Trainer(max_epochs=2000, callbacks=[early_stop_callback,checkpoint_callback],
                                 log_every_n_steps=50)
            trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # 加载验证损失最小的模型权重
            best_model_path = checkpoint_callback.best_model_path
            best_model = liu_attention_Lightning.load_from_checkpoint(best_model_path)

            # 测试
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_inputs = torch.tensor(get_triplets(test_inputs, max_length))
            test_dataset = StructureDataset(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=test_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_worker,
                                     persistent_workers=True)

            predict = trainer.test(model=lightning_model, dataloaders=test_loader)

            mae = predict[0]['test_mae']
            mae_list.append(mae)

        visualize_results(mae_list, dataset_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='liu_attention')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
    parser.add_argument('--gamma', type=float, default=1.8, help='Gamma value for RBF')
    parser.add_argument('--cutoff', type=int, default=3, help='Cutoff length for triplets')
    parser.add_argument('--num_worker', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--patience', type=int, default=300, help='Patience for early stopping')
    parser.add_argument('--embedding_dim', type=int, default=10, help='Embedding dimension for the model')
    parser.add_argument('--num_features', type=int, default=10, help='Number of features for the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the model')
    parser.add_argument('--output_size', type=int, default=1, help='Output size for the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model')
    parser.add_argument('--dropout', type=float, default=0, help='parameter dropout')

    args = parser.parse_args()

    main(args)









