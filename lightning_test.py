from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import os
import random
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_lightning import seed_everything
import argparse

# 加载数据
def load_data_from_folder(folder_path):
    data = []
    labels = []
    for label_type in ['pos', 'neg']:
        path = os.path.join(folder_path, label_type)
        label = 1 if label_type == 'pos' else 0
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                    data.append(file.read())
                    labels.append(label)
    return data, labels

# 预处理文本数据
def preprocess_text(text_data):
    return [simple_preprocess(doc) for doc in text_data]

def text_to_word_vectors(text_data, w2v_module):
    model = w2v_module.wv
    vector_size = w2v_module.vector_size
    word_vectors = []
    for doc in text_data:
        vectors = [model[word] if word in model else np.zeros(vector_size) for word in doc]
        word_vectors.append(vectors)
    return word_vectors


# 填充或截断序列
def pad_sequences(sequences, max_len, vector_size):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_sequences.append(seq[:max_len])
        else:
            padded_sequences.append(seq + [np.zeros(vector_size)] * (max_len - len(seq)))
    return padded_sequences

# 转换为Tensor
def to_tensor(data, batch_size=64):
    tensors = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_np = np.array(batch, dtype=np.float32)
        tensors.append(torch.tensor(batch_np, dtype=torch.float32))
    return torch.cat(tensors)

def labels_to_tensor(labels, batch_size=5000):
    tensors = []
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i + batch_size]
        batch_np = np.array(batch, dtype=np.float32)
        tensors.append(torch.tensor(batch_np, dtype=torch.float32))
    return torch.cat(tensors)


class SentimentDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        input = self.text[index]
        label = self.labels[index]
        return {'input': input, 'labels': label}

class Word2VecDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]
    
# Word2Vec Lightning模块
class Word2VecModule(pl.LightningModule):
    def __init__(self, sentences, vector_size=100, window=5, min_count=2, workers=4):
        super().__init__()
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.wv = None
        
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        pass
    
    def training_step(self, batch, batch_idx):
        pass
    
    def train_dataloader(self):
        dataset = Word2VecDataset(self.sentences)
        return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    def fit(self):
        self.model = Word2Vec(sentences=self.sentences,
                              vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              workers=self.workers)
        self.wv = self.model.wv

    def forward(self, sentences):
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, total_examples=len(sentences), epochs=self.current_epoch + 1)
        return self.model.wv

class SentimentTransformer(pl.LightningModule):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1, learning_rate: float = 2e-4):
        super(SentimentTransformer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.validation_outputs = []
        self.test_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        transformer_out = self.transformer(gru_out)
        pooled_output = transformer_out.mean(dim=1)
        logits = self.fc(pooled_output)
        probs = torch.sigmoid(logits)
        return probs

    def training_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = F.binary_cross_entropy(predictions, label)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = F.binary_cross_entropy(predictions, label)
        preds = torch.round(predictions)
        self.validation_outputs.append({'preds': preds, 'labels': label, 'loss': loss})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        val_f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')
        val_precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        val_recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_precision', val_precision, prog_bar=True)
        self.log('val_recall', val_recall, prog_bar=True)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = F.binary_cross_entropy(predictions, label)
        preds = torch.round(predictions)
        self.test_outputs.append({'preds': preds, 'labels': label, 'loss': loss})
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs])
        labels = torch.cat([x['labels'] for x in self.test_outputs])
        test_f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')
        test_precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        test_recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_precision', test_precision, prog_bar=True)
        self.log('test_recall', test_recall, prog_bar=True)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# 定义模型
class SentimentRNN(pl.LightningModule):
    def __init__(self, vector_size, hidden_dim, output_dim, n_layers, bidirectional, dropout, learning_rate : float = 2e-4):
        super().__init__()
        self.rnn = nn.GRU(vector_size, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.validation_outputs = []
        self.test_outputs = []
        self.save_hyperparameters()
    
    def forward(self, text):
        rnn_output, hidden = self.rnn(text)
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)

    def training_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(predictions, label)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(predictions, label)
        preds = torch.round(torch.sigmoid(predictions))
        self.validation_outputs.append({'preds': preds, 'labels': label, 'loss': loss})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        val_f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')
        val_precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        val_recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_precision', val_precision, prog_bar=True)
        self.log('val_recall', val_recall, prog_bar=True)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        text, label = batch['input'], batch['labels']
        predictions = self.forward(text).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(predictions, label)
        preds = torch.round(torch.sigmoid(predictions))
        self.test_outputs.append({'preds': preds, 'labels': label, 'loss': loss})
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs])
        labels = torch.cat([x['labels'] for x in self.test_outputs])
        test_f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')
        test_precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        test_recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_precision', test_precision, prog_bar=True)
        self.log('test_recall', test_recall, prog_bar=True)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
def main(hparms):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed = hparms.init_seed
    seed_everything(init_seed)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # 加载训练数据
    data, labels = load_data_from_folder(r'./aclImdb/train')

    # 预处理文本数据
    sentences = preprocess_text(data)

    # 训练Word2Vec模型
    w2v_model = Word2VecModule(sentences=sentences, vector_size=hparms.vector_size, window=hparms.window, min_count=2, workers=4)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(w2v_model)
    w2v_model.fit()

    # 获取Word2Vec模型中的词向量
    word_vectors = text_to_word_vectors(sentences, w2v_model)

    # padding
    data = pad_sequences(word_vectors, max_len=hparms.pad_len, vector_size=hparms.vector_size)

    # 转换为Tensor
    data = to_tensor(data)
    labels = labels_to_tensor(labels)

    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=init_seed)

    train_loader = DataLoader(SentimentDataset(train_data, train_labels), batch_size=hparms.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SentimentDataset(val_data, val_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    # model = SentimentRNN(vector_size=hparms.vector_size, hidden_dim=hparms.hidden_dim, output_dim=1, n_layers=hparms.n_layers,
    #                     bidirectional=True, dropout=hparms.dropout, learning_rate=hparms.learning_rate)
    model = SentimentTransformer(num_heads=8, num_layers=8, hidden_dim=hparms.hidden_dim, input_dim=hparms.vector_size)

    early_stopping = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=hparms.patience, verbose=True, mode="max")
    checkpoint = ModelCheckpoint(monitor='val_f1', mode='max', dirpath='checkpoints/', filename='sentiment_rnn-{epoch:02d}-{val_f1:.2f}', save_top_k=1)
    tsb_logger = TensorBoardLogger("tb_logs", name="sentiment_rnn")
    trainer = pl.Trainer(max_epochs=20, callbacks=[early_stopping, checkpoint], logger=tsb_logger, accelerator='gpu', devices=2)
    trainer.fit(model, train_loader, val_loader)

    # 测试
    test_data, test_labels = load_data_from_folder(r'./aclImdb/test')
    test_sentences = preprocess_text(test_data)
    test_word_vectors = text_to_word_vectors(test_sentences, w2v_model)
    test_data = pad_sequences(test_word_vectors, max_len=hparms.pad_len, vector_size=hparms.vector_size)
    test_data = to_tensor(test_data)
    test_labels = labels_to_tensor(test_labels)
    test_loader = DataLoader(SentimentDataset(test_data, test_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    best_model_path = checkpoint.best_model_path
    best_model = SentimentRNN.load_from_checkpoint(best_model_path)
    trainer.test(best_model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment Analysis with PyTorch Lightning')
    parser.add_argument('--init_seed', type=int, default=42, help='Seed for initializing random number generators')
    parser.add_argument('--vector_size', type=int, default=128, help='Dimensionality of the word vectors')
    parser.add_argument('--window', type=int, default=5, help='Context window size for Word2Vec training')
    parser.add_argument('--pad_len', type=int, default=500, help='Maximum length of padded sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimensionality of hidden layers in RNN')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers in RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate for training the model')
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs with no improvement after which training will be stopped')

    args = parser.parse_args()
    main(args)