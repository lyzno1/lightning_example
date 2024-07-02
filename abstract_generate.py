import json
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import pytorch_lightning as pl
import torch
import pytorch_lightning as pl

# 数据预处理函数
def preprocess_data(jsonl_file) -> List[dict]:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 自定义数据集类
class TextSummaryDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: BartTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        article = ' '.join(data_item['article'])
        summary = data_item['summary']

        # 使用BART的tokenizer编码文章和摘要
        encoding = self.tokenizer(article, summary, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'summary': summary
        }

class TextSummaryModel(pl.LightningModule):
    def __init__(self, tokenizer: BartTokenizer, learning_rate: float = 2e-5):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        return self.model.generate(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        summary_ids = self.tokenizer.batch_encode_plus([batch['summary']], max_length=50, truncation=True, padding='max_length', 
                                                       return_tensors='pt')['input_ids']
        outputs = self(input_ids, attention_mask)
        loss = self.model.compute_loss(outputs, summary_ids.view(-1, summary_ids.shape[-1]))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        summary_ids = self.tokenizer.batch_encode_plus([batch['summary']], max_length=50, truncation=True, padding='max_length', 
                                                       return_tensors='pt')['input_ids']
        outputs = self(input_ids, attention_mask)
        loss = self.model.compute_loss(outputs, summary_ids.view(-1, summary_ids.shape[-1]))
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        summary_ids = self.tokenizer.batch_encode_plus([batch['summary']], max_length=50, truncation=True, padding='max_length', 
                                                       return_tensors='pt')['input_ids']
        outputs = self(input_ids, attention_mask)
        loss = self.model.compute_loss(outputs, summary_ids.view(-1, summary_ids.shape[-1]))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)

def main():
    # 加载和预处理数据
    data = preprocess_data('../final/train.simple.label.jsonl')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    dataset = TextSummaryDataset(data, tokenizer)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化模型
    model = TextSummaryModel(tokenizer)

    # 设置训练参数
    trainer = pl.Trainer(max_epochs=5)

    # 训练模型
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    pl.seed_everything(42)
    main()