import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import yeojohnson, yeojohnson_normmax, yeojohnson_llf
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
import argparse
from sklearn.preprocessing import PowerTransformer

# 数据预处理
def preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='gbk')
    
    data['entry_date'] = pd.to_datetime(data['entry_date'])
    data['departure_date'] = pd.to_datetime(data['departure_date'])
    
    data['duration'] = (data['departure_date'] - data['entry_date']).dt.days
    
    pt = PowerTransformer(method='yeo-johnson')
    data['CO2排放总量_transformed'] = pt.fit_transform(data['CO2排放总量'].values.reshape(-1, 1)) # 进行Yeo-Johnson变换
    
    data = data.sort_values(by='duration', ascending=False)
    
    return data, pt

# 自定义数据集
class EmissionsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = data[['duration']].values
        self.labels = data['CO2排放总量_transformed'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

# pl数据模块
class EmissionsDataModule(pl.LightningDataModule):
    def __init__(self, hparams, df, pt):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers
        self.df = df
        self.pt = pt
        self.init_seed = hparams.init_seed

    def setup(self, stage=None): # 划分数据集
        dataset = EmissionsDataset(self.df)

        train_and_val_dataset, self.test_dataset = train_test_split(dataset, test_size=0.1, random_state=self.init_seed)
        self.train_dataset, self.val_dataset = train_test_split(train_and_val_dataset, test_size=0.1, random_state=self.init_seed)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


# RNN 
class RNNModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# GRU 结合 Transformer
class GRUModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2)
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        at_out, _ = self.attention(out, out, out)
        out = self.ln(out + at_out)
        out = self.ffn(out)
        out = self.ln(out + at_out)
        out = self.fc(out[:, -1, :])
        return out

# LSTM
class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, input_dim=1, output_dim=1, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出，并送入全连接层
        out = self.fc(out[:, -1, :])
        return out

# 全连接层(Linear)
class FullyConnected(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=1, output_dim=1, is_linear=False):
        super(FullyConnected, self).__init__()
        self.is_linear = is_linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.only = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        if self.is_linear:
            x = self.only(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        return x

# pytorch lightning 模块
class EmissionsPredictor(pl.LightningModule):
    def __init__(self, hparams, pt):
        super(EmissionsPredictor, self).__init__()
        self.save_hyperparameters(hparams) # 保存超参数
        self.pt = pt
        self.model_type = hparams.model_type

        if self.model_type == 'RNN':
            self.model = RNNModel(hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers)
        elif self.model_type == 'GRU':
            self.model = GRUModel(hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers)
        elif self.model_type == 'LSTM':
            self.model = LSTMModel(hidden_dim=hparams.hidden_dim, num_layers=hparams.num_layers, bidirectional=hparams.bidirectional)
        elif self.model_type == 'FC':
            self.model = FullyConnected(hidden_dim=hparams.hidden_dim, is_linear=False)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.criterion = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss' : loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {'loss' : loss}

    def test_step(self, batch, batch_idx):
        features, labels_transformed = batch
        outputs = self(features).squeeze()
        
        # 逆变换
        original_predictions = self.pt.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
        original_labels = self.pt.inverse_transform(labels_transformed.cpu().numpy().reshape(-1, 1))
        
        # 计算原始尺度的MAE
        mae = mean_absolute_error(original_labels, original_predictions)
        
        self.log('test_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        
        return mae

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=hparams.lr)

def main(hparams):
    seed_everything(hparams.init_seed) # 固定随机种子

    df, pt = preprocess_data(hparams.data_path) # 预处理数据并保存变换器

    model = EmissionsPredictor(hparams=hparams, pt=pt)
    
    # Trainer
    earlystopping = EarlyStopping('val_loss', patience=hparams.patience, verbose=True, min_delta=0.00, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          dirpath='emissions',
                                          filename='emissions-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=1)
    progress_bar = RichProgressBar()

    callback = [earlystopping, checkpoint_callback, progress_bar]
    trainer = pl.Trainer(max_epochs=hparams.max_epochs, callbacks=callback, accelerator='gpu', devices=1, logger=True)
    
    # DataModule
    dm = EmissionsDataModule(hparams=hparams, df=df, pt=pt)

    trainer.fit(model, datamodule=dm)

    best_model_path = trainer.checkpoint_callback.best_model_path

    # 测试
    trainer.test(
        model=model,
        datamodule=dm,
        ckpt_path=best_model_path
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Carbon dioxide emissions with PyTorch Lightning')
    parser.add_argument('--init_seed', type=int, default=42, help='Seed for initializing random number generators')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimensionality of hidden layers in RNN')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning Rate for training the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads in Multi-Head Attention')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train the model')
    parser.add_argument('--data_path', type=str, default='data.csv', help='Path to the CSV data file')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['RNN', 'GRU', 'LSTM', 'FC'], help='Type of model to use')
    parser.add_argument('--bidirectional', type=bool, default=True, help='Whether to use a bidirectional LSTM')

    hparams = parser.parse_args()
    main(hparams)
