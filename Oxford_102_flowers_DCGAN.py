import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

LATENT_DIM = 100
IMAGE_SIZE = 64
N_CHANNELS = 3

G_HIDDEN_SIZE = 64
D_HIDDEN_SIZE = 64

LEARN_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
BATCH_SIZE = 128

def weights_init(model):
    name = model.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    if "BatchNorm" in name:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_features, out_features):
            layers = [nn.ConvTranspose2d(in_features, out_features, 4,
                                         stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            return layers

        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, G_HIDDEN_SIZE * 8, 4,
                               bias=False),
            nn.BatchNorm2d(G_HIDDEN_SIZE * 8),
            nn.ReLU(inplace=True),
            *block(G_HIDDEN_SIZE * 8, G_HIDDEN_SIZE * 4),
            *block(G_HIDDEN_SIZE * 4, G_HIDDEN_SIZE * 2),
            *block(G_HIDDEN_SIZE * 2, G_HIDDEN_SIZE),
            nn.ConvTranspose2d(G_HIDDEN_SIZE, N_CHANNELS, 4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_features, out_features):
            layers = [nn.Conv2d(in_features, out_features, 4,
                                stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(out_features),
                      nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.net = nn.Sequential(
            nn.Conv2d(N_CHANNELS, D_HIDDEN_SIZE, 4,
                      stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            *block(D_HIDDEN_SIZE, D_HIDDEN_SIZE * 2),
            *block(D_HIDDEN_SIZE * 2, D_HIDDEN_SIZE * 4),
            *block(D_HIDDEN_SIZE * 4, D_HIDDEN_SIZE * 8),
            nn.Conv2d(D_HIDDEN_SIZE * 8, 1, 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

class GANModel(pl.LightningModule):
    def __init__(self):
        super(GANModel, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()
        
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.adversarial_loss = nn.BCELoss()
        self.G_image = None

    def forward(self, noise):
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        image, _ = batch
        batch_size = image.size(0)
        real = torch.ones(batch_size,)
        fake = torch.zeros(batch_size,)

        if optimizer_idx == 0:
            random_noise = torch.rand(batch_size, LATENT_DIM, 1, 1)
            self.G_image = self.generator(random_noise)

            D_pred = self.discriminator(self.G_image)
            loss = self.adversarial_loss(D_pred, real)
            self.log("generator loss", loss)

        if optimizer_idx == 1:
            D_pred_real = self.discriminator(image)
            loss_real = self.adversarial_loss(D_pred_real, real)
            
            self.G_image = self.G_image.detach()
            D_pred_fake = self.discriminator(self.G_image)
            loss_fake = self.adversarial_loss(D_pred_fake, fake)
            loss = (loss_real + loss_fake) / 2
            self.log("discriminator loss", loss)

        return loss

    def configure_optimizers(self):
        return [Adam(self.generator.parameters(),
                     lr=LEARN_RATE, betas=(BETA1, BETA2)),
                Adam(self.discriminator.parameters(),
                     lr=LEARN_RATE, betas=(BETA1, BETA2))]

    def train_dataloader(self):
        process = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),    
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = ImageFolder(os.getcwd() + '/flower_data',
                                    transform=process)
        return DataLoader(train_dataset, batch_size=128, shuffle=True)


model = GANModel()

wandb_logger = WandbLogger(name="Training", project="Oxford 102 flowers DCGAN")
trainer = pl.Trainer(logger=wandb_logger, max_epochs=5, log_every_n_steps=10,
                     progress_bar_refresh_rate=1)

trainer.fit(model)
