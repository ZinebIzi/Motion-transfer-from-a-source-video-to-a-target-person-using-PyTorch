
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid to output probability
        )


    def forward(self, input):
        return self.model(input)
    
import torch.autograd as autograd

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
    Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((128, 128)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG.load_state_dict(torch.load(self.filename))  # Load the state_dict into netG
            self.netG.eval()


    

    def gradient_penalty(self, discriminator, real_data, fake_data, lambda_gp=10):
        """Calculates the gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_data.device)  # Uniform random number for interpolation
        interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

        d_interpolated = discriminator(interpolated)
        gradients = autograd.grad(
            outputs=d_interpolated, inputs=interpolated,
            grad_outputs=torch.ones(d_interpolated.size(), device=real_data.device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)  # L2 norm of gradients
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def train(self, n_epochs=20, lr=0.0002):
        # Define the optimizers (no need for BCELoss with WGAN-GP)
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            for i, (ske, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)

                # Move data to the same device as the model
                ske, real_images = ske.to(self.netD.device), real_images.to(self.netD.device)
                
                # 1. Train the Discriminator with WGAN-GP loss
                self.netD.zero_grad()

                # Generate fake images from the skeleton input
                fake_images = self.netG(ske)
                
                # Discriminator loss on real and fake images
                outputs_real = self.netD(real_images)
                outputs_fake = self.netD(fake_images.detach())
                lossD_real = -torch.mean(outputs_real)      # Minimize -D(x) for real images
                lossD_fake = torch.mean(outputs_fake)       # Maximize D(G(z)) for fake images

                # Compute gradient penalty
                gp = self.gradient_penalty(self.netD, real_images, fake_images.detach())

                # Total discriminator loss
                lossD = lossD_real + lossD_fake + gp
                lossD.backward()
                optimizerD.step()

                # 2. Train the Generator every 2nd batch
                if i % 2 == 0:
                    self.netG.zero_grad()

                    # Generator loss (maximize the discriminator's response for generated images)
                    outputs = self.netD(fake_images)
                    lossG = -torch.mean(outputs)  # Minimize -D(G(z))
                    lossG.backward()

                    # Update the Generator
                    optimizerG.step()

                # Print statistics
                if i % 50 == 0:
                    print(f"[{epoch + 1}/{n_epochs}][{i}/{len(self.dataloader)}] "
                        f"Loss_D: {lossD:.4f} Loss_G: {lossG:.4f}")

        # Save the model after training
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.netG.state_dict(), self.filename)
        print("Training complete. Model saved.")





    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(500) #5) #200) #4
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

