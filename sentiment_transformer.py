import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
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

# 创建词典并将文本转换为索引
def create_vocab_and_encode(text_data):
    vocab = {}
    encoded_data = []
    idx = 1
    for doc in text_data:
        encoded_doc = []
        for word in doc:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
            encoded_doc.append(vocab[word])
        encoded_data.append(encoded_doc)
    return vocab, encoded_data

# 填充或截断序列
def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_sequences.append(seq[:max_len])
        else:
            padded_sequences.append(seq + [0] * (max_len - len(seq)))
    return padded_sequences

# 转换为Tensor
def to_tensor(data, batch_size=64):
    tensors = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_np = np.array(batch, dtype=np.int64)
        tensors.append(torch.tensor(batch_np, dtype=torch.long))
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SentimentTransformer(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, output_dim=1, max_len=500, learning_rate: float = 2e-4):
        super(SentimentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.validation_outputs = []
        self.test_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        transformer_out = self.transformer(x)
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
        preds = torch.cat([x['preds'] for x in self.validation_outputs]).cpu().float()
        labels = torch.cat([x['labels'] for x in self.validation_outputs]).cpu().float()
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
        preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu().float()
        labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu().float()
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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def main(hparms):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed = hparms.init_seed
    seed_everything(init_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # 加载训练数据
    data_folder = './aclImdb/train'
    data, labels = load_data_from_folder(data_folder)

    # 预处理文本数据
    sentences = preprocess_text(data)

    # 创建词典并将文本转换为索引
    vocab, encoded_data = create_vocab_and_encode(sentences)

    max_len = hparms.pad_len
    padded_sequences = pad_sequences(encoded_data, max_len)
    
    data = to_tensor(padded_sequences)
    labels = labels_to_tensor(labels)

    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=init_seed)

    train_loader = DataLoader(SentimentDataset(train_data, train_labels), batch_size=hparms.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SentimentDataset(val_data, val_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    # 初始化SentimentTransformer模型
    model = SentimentTransformer(
        vocab_size=len(vocab) + 1,
        embedding_dim=hparms.vector_size,
        num_heads=hparms.num_heads,
        num_layers=hparms.num_layers,
        hidden_dim=hparms.hidden_dim,
        max_len=max_len,
        learning_rate=hparms.learning_rate
    )

    # 配置日志记录和检查点
    early_stopping = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=hparms.patience, verbose=True, mode="max")
    checkpoint = ModelCheckpoint(monitor='val_f1', mode='max', dirpath='checkpoints/', filename='sentiment_transformer-{epoch:02d}-{val_f1:.2f}', save_top_k=1)
    tsb_logger = TensorBoardLogger("tb_logs", name="sentiment_transformer")
    trainer = pl.Trainer(max_epochs=hparms.max_epochs, callbacks=[early_stopping, checkpoint], logger=tsb_logger, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)

    # 测试
    test_data_folder = './aclImdb/test'
    test_data, test_labels = load_data_from_folder(test_data_folder)
    test_sentences = preprocess_text(test_data)
    test_encoded_data = [[vocab.get(word, 0) for word in doc] for doc in test_sentences]
    test_padded_sequences = pad_sequences(test_encoded_data, max_len)
    test_data = to_tensor(test_padded_sequences)
    test_labels = labels_to_tensor(test_labels)
    test_loader = DataLoader(SentimentDataset(test_data, test_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    best_model_path = checkpoint.best_model_path
    best_model = SentimentTransformer.load_from_checkpoint(
        best_model_path,
        vocab_size=len(vocab) + 1,
        embedding_dim=hparms.vector_size,
        num_heads=hparms.num_heads,
        num_layers=hparms.num_layers,
        hidden_dim=hparms.hidden_dim,
        max_len=max_len,
        learning_rate=hparms.learning_rate
    )
    trainer.test(best_model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_seed', type=int, default=42, help='Initial seed for randomness')
    parser.add_argument('--vector_size', type=int, default=128, help='Dimension of word vectors')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the transformer model')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in the transformer model')
    parser.add_argument('--pad_len', type=int, default=500, help='Maximum length of padded sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for the optimizer')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    hparms = parser.parse_args()
    main(hparms)
