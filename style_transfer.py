from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            self._build_downsampling_block(64, 128),
            self._build_downsampling_block(128, 256),
            *[ResidualBlock(256) for _ in range(n_residual_blocks)],
            self._build_upsampling_block(256, 128),
            self._build_upsampling_block(128, 64),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def _build_downsampling_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def _build_upsampling_block(self, in_features, out_features):
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._build_discriminator_block(input_nc, 64, normalization=False),
            self._build_discriminator_block(64, 128),
            self._build_discriminator_block(128, 256),
            self._build_discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        )

    def _build_discriminator_block(self, in_features, out_features, normalization=True):
        layers = [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CycleGAN(pl.LightningModule):
    def __init__(self, input_nc, output_nc, lr=2e-4, beta1=0.5, beta2=0.999, lambda_cyc=10.0):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        self.generator_G = Generator(input_nc, output_nc)
        self.generator_F = Generator(output_nc, input_nc)
        self.discriminator_X = Discriminator(input_nc)
        self.discriminator_Y = Discriminator(output_nc)
        self.loss_GAN = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.validation_outputs = []

    def forward(self, x):
        return self.generator_G(x)

    def configure_optimizers(self):
        opt_G = optim.Adam(
            list(self.generator_G.parameters()) + list(self.generator_F.parameters()),
            lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2)
        )
        opt_D = optim.Adam(
            list(self.discriminator_X.parameters()) + list(self.discriminator_Y.parameters()),
            lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2)
        )
        return [opt_G, opt_D], []

    def adversarial_loss(self, y_pred, y_true):
        return self.loss_GAN(y_pred, y_true)

    def cycle_loss(self, y_pred, y_true):
        return self.loss_cycle(y_pred, y_true) * self.hparams.lambda_cyc

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_X, real_Y = batch

        # Generators G and F
        if optimizer_idx == 0:
            fake_Y = self.generator_G(real_X)
            fake_X = self.generator_F(real_Y)
            rec_X = self.generator_F(fake_Y)
            rec_Y = self.generator_G(fake_X)
            loss_GAN_G = self.adversarial_loss(self.discriminator_Y(fake_Y), torch.ones_like(self.discriminator_Y(fake_Y)))
            loss_GAN_F = self.adversarial_loss(self.discriminator_X(fake_X), torch.ones_like(self.discriminator_X(fake_X)))
            loss_cycle_X = self.cycle_loss(rec_X, real_X)
            loss_cycle_Y = self.cycle_loss(rec_Y, real_Y)
            loss_G = loss_GAN_G + loss_GAN_F + loss_cycle_X + loss_cycle_Y
            self.log('loss_G', loss_G, prog_bar=True, on_epoch=True)
            return loss_G

        # Discriminators X and Y
        if optimizer_idx == 1:
            fake_Y = self.generator_G(real_X).detach()
            fake_X = self.generator_F(real_Y).detach()
            loss_D_X_real = self.adversarial_loss(self.discriminator_X(real_X), torch.ones_like(self.discriminator_X(real_X)))
            loss_D_X_fake = self.adversarial_loss(self.discriminator_X(fake_X), torch.zeros_like(self.discriminator_X(fake_X)))
            loss_D_Y_real = self.adversarial_loss(self.discriminator_Y(real_Y), torch.ones_like(self.discriminator_Y(real_Y)))
            loss_D_Y_fake = self.adversarial_loss(self.discriminator_Y(fake_Y), torch.zeros_like(self.discriminator_Y(fake_Y)))
            loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5
            loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
            self.log('loss_D_X', loss_D_X, prog_bar=True, on_epoch=True)
            self.log('loss_D_Y', loss_D_Y, prog_bar=True, on_epoch=True)
            return loss_D_X + loss_D_Y

    def validation_step(self, batch, batch_idx):
        real_X, real_Y = batch
        fake_Y = self.generator_G(real_X)
        fake_X = self.generator_F(real_Y)
        rec_X = self.generator_F(fake_Y)
        rec_Y = self.generator_G(fake_X)
        loss_GAN_G = self.adversarial_loss(self.discriminator_Y(fake_Y), torch.ones_like(self.discriminator_Y(fake_Y)))
        loss_GAN_F = self.adversarial_loss(self.discriminator_X(fake_X), torch.ones_like(self.discriminator_X(fake_X)))
        loss_cycle_X = self.cycle_loss(rec_X, real_X)
        loss_cycle_Y = self.cycle_loss(rec_Y, real_Y)
        loss_G = loss_GAN_G + loss_GAN_F + loss_cycle_X + loss_cycle_Y
        self.validation_outputs.append({'loss_G': loss_G})

    def on_validation_epoch_end(self):
        avg_loss_G = torch.stack([x['loss_G'] for x in self.validation_outputs]).mean()
        self.log('val_loss_G', avg_loss_G, prog_bar=True)
        self.validation_outputs = []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_X = ImageFolder(os.path.join('data', 'trainA'), transform=transform)
        dataset_Y = ImageFolder(os.path.join('data', 'trainB'), transform=transform)
        dataset = CombinedDataset(dataset_X, dataset_Y)
        return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_X = ImageFolder(os.path.join('data', 'valA'), transform=transform)
        dataset_Y = ImageFolder(os.path.join('data', 'valB'), transform=transform)
        dataset = CombinedDataset(dataset_X, dataset_Y)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

class CombinedDataset(Dataset):
    def __init__(self, dataset_X, dataset_Y):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y

    def __len__(self):
        return min(len(self.dataset_X), len(self.dataset_Y))

    def __getitem__(self, index):
        item_X = self.dataset_X[index % len(self.dataset_X)]
        item_Y = self.dataset_Y[index % len(self.dataset_Y)]
        return item_X[0], item_Y[0]

def main():
    logger = TensorBoardLogger("tb_logs", name="CycleGAN")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_G',
        dirpath='checkpoints',
        filename='CycleGAN-{epoch:02d}-{val_loss_G:.2f}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor='val_loss_G',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    model = CycleGAN(input_nc=3, output_nc=3)
    trainer = pl.Trainer(max_epochs=200, gpus=1, logger=logger, callbacks=[checkpoint_callback, lr_monitor, early_stop_callback])
    trainer.fit(model)

if __name__ == '__main__':
    main()
