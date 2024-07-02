import pandas as pd
import re
import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from transformers import BertTokenizer
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score

# 文本预处理函数
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# 微博Dataset类
class WeiboDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment = self.data.iloc[idx]['comment']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'comment_text': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# BERT分类模型
class WeiboCommentClassifier(pl.LightningModule):
    def __init__(self, n_classes, learning_rate=2e-5):
        super(WeiboCommentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(p=0)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.learning_rate = learning_rate
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = output['pooler_output']
        output = self.dropout(pooled_output)
        return self.classifier(output)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.validation_outputs.append({'preds': preds, 'labels': labels, 'loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        val_f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        self.log('val_f1', val_f1, prog_bar=True)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.validation_outputs = []

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.test_outputs.append({'preds': preds, 'labels': labels, 'loss': loss})
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs])
        labels = torch.cat([x['labels'] for x in self.test_outputs])
        test_f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def set_random_seed(seed): # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    init_seed = 42
    set_random_seed(init_seed)

    # 读取数据集
    data = pd.read_csv('C:/Users/lyz/Desktop/weibo_comments.csv')
    data = data[['comment', 'label']]

    # 预处理所有评论
    data['comment'] = data['comment'].apply(preprocess_text)

    # 使用BertTokenizer进行分词
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', force_download=True)

    # 定义数据集和DataLoader
    MAX_LEN = 32
    BATCH_SIZE = 64

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 存储每个折的结果
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f'Fold {fold + 1}')

        train_val_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

        train_dataset = WeiboDataset(train_data, tokenizer, MAX_LEN)
        val_dataset = WeiboDataset(val_data, tokenizer, MAX_LEN)
        test_dataset = WeiboDataset(test_data, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # 设置模型检查点回调
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            dirpath=None ,
            filename=f'best-checkpoint/fold_{fold + 1}/bert-base-chinese-weibo-comment-classifier-{fold + 1}',
            save_top_k=1,
            mode='max'
        )

        # 初始化分类模型
        model = WeiboCommentClassifier(n_classes=2)

        # 初始化PyTorch Lightning训练器
        trainer = pl.Trainer(
            max_epochs=20,
            callbacks=[checkpoint_callback]
        )

        # 训练模型
        trainer.fit(model, train_loader, val_loader)

        # 测试模型
        test_result = trainer.test(model, test_loader)
        fold_results.append(test_result)

    # 计算平均性能指标
    avg_test_f1 = np.mean([result[0]['test_f1'] for result in fold_results])
    avg_test_loss = np.mean([result[0]['test_loss'] for result in fold_results])

    print(f'Average Test F1 Score: {avg_test_f1}')
    print(f'Average Test Loss: {avg_test_loss}')
    for fold, result in enumerate(fold_results, 1):
        print(f'Fold {fold} - Test F1: {result[0]["test_f1"]}, Test Loss: {result[0]["test_loss"]}')
