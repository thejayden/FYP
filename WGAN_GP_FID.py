from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
from cv2 import GC_EVAL
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
from torch.autograd import Variable
import torch.autograd as autograd

from scipy import linalg
from torchvision.utils import save_image

def main():
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    print(torch.cuda.is_available())

    # !git clone -b processed_hyperbolas https://github.com/thejayden/gpr_field_data.git

    path = os.path.join(os.getcwd(), 'train')

    # Root directory for dataset
    dataroot = path
    # Number of workers for dataloader
    workers = 2
    # Batch size during training
    # batch_size = 128
    batch_size = 500
    # batch_size = 1
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 400
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    num_epochs = 500
    # number of training steps for discriminator per iter
    n_critic = 1
    
    # Learning rate for optimizers
    # lr = 0.0002
    lr = 0.00005

    g_lr = 0.0001
    d_lr = 0.0004

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # # Number of GPUs available. Use 0 for CPU mode.
    # ngpu = 0

    # Loss weight for gradient penalty
    lambda_gp = 10

    k = 2
    p = 6

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                #  transforms.Resize(image_size),
                                #  transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    cuda = True if torch.cuda.is_available() else False
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    print("HOW MANY CUDA?:" + str(torch.cuda.device_count()))
    # device = 'cuda'
    print("IS CUDA?: "+ str(torch.cuda.is_available()))

    # Show a random image
    idx = np.random.choice(len(dataset)) # get a random index in range [0, len(dataset))
    # img, label = dataset[idx]
    img, label = dataset[idx]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.savefig('./images/sample.png')

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            # self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*8 = 2048) x 4 x 4

                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*4 = 1024) x 8 x 8

                nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*2 = 512) x 16 x 16

                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf = 256) x 32 x 32

                nn.ConvTranspose2d( ngf * 2, int(ngf), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(ngf)),
                nn.ReLU(True),
                # state size. (ngf/2 = 128) x 64 x 64

                nn.ConvTranspose2d( int(ngf), int(ngf/2), 4, 2, 1, bias=False),
                nn.BatchNorm2d(int(ngf/2)),
                nn.ReLU(True),
                # state size. (ngf/4 = 64) x 128 x 128

                nn.ConvTranspose2d( int(ngf/2), nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc = 3) x 256 x 256
            )

        def forward(self, input):
            return self.main(input)

    # Create the generator
    netG = Generator()
    netG = nn.DataParallel(netG)
    netG = Generator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # # Print the model
    print(netG)
    # summary(netG, (400, 1, 1))

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            # self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, int(ndf/2), 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(int(ndf/2), ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    # Create the Discriminator
    netD = Discriminator()
    netD = nn.DataParallel(netD)
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)
    # summary(netD, (3, 256, 256))

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Establish convention for real and fake labels during training
    # real_label = 1
    # fake_label = 0
    """adding label smoothing"""
    real_label=0.9
    fake_label=0.1

    # Setup Adam optimizers for both G and D
    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, 0.999))

    class InceptionV3(nn.Module):
        """Pretrained InceptionV3 network returning feature maps"""

        # Index of default block of inception to return,
        # corresponds to output of final average pooling
        DEFAULT_BLOCK_INDEX = 3

        # Maps feature dimensionality to their output blocks indices
        BLOCK_INDEX_BY_DIM = {
            64: 0,   # First max pooling features
            192: 1,  # Second max pooling featurs
            768: 2,  # Pre-aux classifier features
            2048: 3  # Final average pooling features
        }

        def __init__(self,
                    output_blocks=[DEFAULT_BLOCK_INDEX],
                    resize_input=True,
                    normalize_input=True,
                    requires_grad=False):
            
            super(InceptionV3, self).__init__()

            self.resize_input = resize_input
            self.normalize_input = normalize_input
            self.output_blocks = sorted(output_blocks)
            self.last_needed_block = max(output_blocks)

            assert self.last_needed_block <= 3, \
                'Last possible output block index is 3'

            self.blocks = nn.ModuleList()

            
            inception = models.inception_v3(pretrained=True)

            # Block 0: input to maxpool1
            block0 = [
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block0))

            # Block 1: maxpool1 to maxpool2
            if self.last_needed_block >= 1:
                block1 = [
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2)
                ]
                self.blocks.append(nn.Sequential(*block1))

            # Block 2: maxpool2 to aux classifier
            if self.last_needed_block >= 2:
                block2 = [
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                ]
                self.blocks.append(nn.Sequential(*block2))

            # Block 3: aux classifier to final avgpool
            if self.last_needed_block >= 3:
                block3 = [
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
                ]
                self.blocks.append(nn.Sequential(*block3))

            for param in self.parameters():
                param.requires_grad = requires_grad

        def forward(self, inp):
            """Get Inception feature maps
            Parameters
            ----------
            inp : torch.autograd.Variable
                Input tensor of shape Bx3xHxW. Values are expected to be in
                range (0, 1)
            Returns
            -------
            List of torch.autograd.Variable, corresponding to the selected output
            block, sorted ascending by index
            """
            outp = []
            x = inp

            if self.resize_input:
                x = F.interpolate(x,
                                size=(299, 299),
                                mode='bilinear',
                                align_corners=False)

            if self.normalize_input:
                x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

            for idx, block in enumerate(self.blocks):
                x = block(x)
                if idx in self.output_blocks:
                    outp.append(x)

                if idx == self.last_needed_block:
                    break

            return outp
        
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model=model.cuda()

    def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                        cuda=True):
        model.eval()
        act=np.empty((len(images), dims))
        
        if cuda:
            batch=images.cuda()
        else:
            batch=images
        pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
        
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_frechet(images_real,images_fake,model):
        mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
        mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
        
        """get fretched distance"""
        fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value

    print("Generator Parameters:",sum(p.numel() for p in netG.parameters() if p.requires_grad))
    print("Discriminator Parameters:",sum(p.numel() for p in netD.parameters() if p.requires_grad))

    def compute_gradient_penalty(D, real_samples, fake_samples):
    # """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    img_list = []
    G_losses = []
    D_losses = []
    G_E_losses = []
    D_E_losses = []
    iters = 0
    batches_done = 0

    print("Starting Training Loop...")
    
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, imgs in enumerate(dataloader, 0):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], nz))))

            # Generate a batch of images
            fake_imgs = netG(z)

            # Real images
            real_validity = netD(real_imgs)
            # Fake images
            fake_validity = netD(fake_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            D_losses.append(d_loss.item())   

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = netG(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = netD(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizerG.step()

                G_losses.append(g_loss.item())

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [iters: %d] [D loss: %f] [G loss: %f]"
                    % (epoch, num_epochs, i, len(dataloader), iters, d_loss.item(), g_loss.item())
                )

                # if batches_done % sample_interval == 0:
                #     save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += n_critic

                iters += 1
        
        frechet_dist=calculate_frechet(real_imgs,fake_imgs,model)    
            
        if ((epoch+1)%5==0):
            
            G_E_losses.append(g_loss.item())
            D_E_losses.append(d_loss.item())   

            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFrechet_Distance: %.4f'
                        % (epoch+1, num_epochs,
                            d_loss.item(), g_loss.item(),frechet_dist))
                      
            # save_image(fake_imgs.data[:25], "images/%d.png" % epoch+1, nrow=5, normalize=True)
            # plt.savefig('./images/' + str(epoch+1) + '.png')

            for i in range (25):
                image = fake_imgs[:i]
                dir = "images/epoch" + str(epoch+1)
                try:
                    os.makedirs(dir)
                except FileExistsError:
                    # directory already exists
                    pass
                save_image(image, dir + '/' + str(i) + '.png')                                

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('./images/GD_Loss.png')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss For Each Epoch")
    plt.plot(G_E_losses,label="G")
    plt.plot(D_E_losses,label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('./images/GD_E_Loss.png')

    # print("Length of img_list:" + str(len(img_list)))
    # print(img_list)

if __name__ == '__main__':
    main()