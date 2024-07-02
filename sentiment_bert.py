import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import random
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_lightning import seed_everything
import argparse

# 加载数据
def load_data_and_labels(folder_path):
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
def preprocess_text(tokenizer, text_data, max_len):
    encodings = tokenizer(text_data, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    }
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]  # Assuming 'labels' is a key in your labels tensor
        return item


class SentimentBERT(pl.LightningModule):
    def __init__(self, learning_rate, num_labels=2):
        super(SentimentBERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.learning_rate = learning_rate
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.validation_outputs.append({'preds': preds, 'labels': batch['labels'], 'loss': loss})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs]).cpu()
        labels = torch.cat([x['labels'] for x in self.validation_outputs]).cpu()
        val_f1 = f1_score(labels, preds, average='binary')
        val_precision = precision_score(labels, preds, average='binary')
        val_recall = recall_score(labels, preds, average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_precision', val_precision, prog_bar=True)
        self.log('val_recall', val_recall, prog_bar=True)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        self.test_outputs.append({'preds': preds, 'labels': batch['labels'], 'loss': loss})
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu()
        labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu()
        test_f1 = f1_score(labels, preds, average='binary')
        test_precision = precision_score(labels, preds, average='binary')
        test_recall = recall_score(labels, preds, average='binary')
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_precision', test_precision, prog_bar=True)
        self.log('test_recall', test_recall, prog_bar=True)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

def main(hparms):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed = hparms.init_seed
    seed_everything(init_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)

    # 加载训练数据
    data_folder = './aclImdb/train'
    data, labels = load_data_and_labels(data_folder)

    # 预处理文本数据
    input_ids, attention_mask = preprocess_text(tokenizer, data, hparms.pad_len)

    # 划分训练集和验证集
    train_data, val_data, train_masks, val_masks, train_labels, val_labels = train_test_split(input_ids, attention_mask, labels, test_size=0.2, random_state=init_seed)
    train_loader = DataLoader(SentimentDataset(train_data, train_labels), batch_size=hparms.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SentimentDataset(val_data, val_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    # 初始化SentimentBERT模型
    model = SentimentBERT(learning_rate=hparms.learning_rate)

    # 配置日志记录和检查点
    early_stopping = EarlyStopping(monitor="val_f1", min_delta=0.00, patience=hparms.patience, verbose=True, mode="max")
    checkpoint = ModelCheckpoint(monitor='val_f1', mode='max', dirpath='checkpoints/', filename='sentiment_bert-{epoch:02d}-{val_f1:.2f}', save_top_k=1)
    tsb_logger = TensorBoardLogger("tb_logs", name="sentiment_bert")
    trainer = pl.Trainer(max_epochs=hparms.max_epochs, callbacks=[early_stopping, checkpoint], logger=tsb_logger, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)

    # 测试
    test_data_folder = './aclImdb/test'
    test_data, test_labels = load_data_and_labels(test_data_folder)
    test_encodings = preprocess_text(tokenizer, test_data, hparms.pad_len)

    test_data = test_encodings
    test_labels = torch.tensor(test_labels)
    test_loader = DataLoader(SentimentDataset(test_data, test_labels), batch_size=hparms.batch_size, shuffle=False, num_workers=0)

    best_model_path = checkpoint.best_model_path
    best_model = SentimentBERT.load_from_checkpoint(best_model_path)
    trainer.test(best_model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_seed', type=int, default=42, help='Initial seed for randomness')
    parser.add_argument('--pad_len', type=int, default=512, help='Maximum length of padded sequences')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')
    parser.add_argument('--max_epochs', type=int, default=3, help='Maximum number of epochs for training')
    parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')

    hparms = parser.parse_args()
    main(hparms)
