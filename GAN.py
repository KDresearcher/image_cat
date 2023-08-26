import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import cv2
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="训练次数")
parser.add_argument("--batch_size", type=int, default=1024, help="数量为训练数据的2倍，1倍，1/2或者1/4或者1/8")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--lambda_gp", type=int, default=10, help="gradient_penalty")
parser.add_argument("--save_iteration", type=int, default=50, help="每隔多少次输出一次")
parser.add_argument("--loss", type=str, default='lecam', help="选择哪种损失，有wgan_gp和lecam")
parser.add_argument("--type", type=str, default='agg', help='选择agg或者ce')

opt = parser.parse_args()


img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, opt.latent_dim * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.2)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
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

# ----------
#  Training
# ----------


def train():
    transform = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize(
        [0.5 for _ in range(3)],
        [0.5 for _ in range(3)]
    )])

    if opt.type == 'agg':
        os.makedirs("dataset/aggregate_small", exist_ok=True)
        data = ImageFolder(root='./dataset/aggregate_small', transform=transform)

    elif opt.type == 'ce':
        os.makedirs("dataset/cement_small", exist_ok=True)
        data = ImageFolder(root='./dataset/cement_small', transform=transform)

    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
    d_pre = []
    g_pre = []
    d_los = []
    g_los = []
    ema_real = ema_fake = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            for _ in range(opt.n_critic):
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                fake_imgs = generator(z)

                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs)

                if opt.loss == 'wgan_gp':
                    # wgan_gp
                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
                elif opt.loss == 'lecam':
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
                             0.3 * (torch.mean(F.relu(real_validity - ema_fake).pow(2))
                                     + torch.mean(F.relu(ema_real - fake_validity).pow(2)))

                d_loss.backward()
                optimizer_D.step()
                if opt.loss == 'lecam':
                    ema_real = torch.tensor(real_validity, requires_grad=False)
                    ema_fake = torch.tensor(fake_validity, requires_grad=False)

            optimizer_G.zero_grad()

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)

            # wgan_gp
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), torch.mean(real_validity).item(), torch.mean(fake_validity).item()))
        d_pre.append(torch.mean(real_validity).item())
        g_pre.append(torch.mean(fake_validity).item())
        d_los.append(d_loss.item())
        g_los.append(g_loss.item())

        if epoch % opt.save_iteration == 0:
            os.makedirs('./save_image/%s/lecam_image/%s/' % (opt.type, str(epoch)), exist_ok=True)
            generator.eval()
            for p in range(100):
                noise = Variable(Tensor(np.random.normal(0, 1, (1, 128))))
                fake = generator(noise)
                fake = fake.mul(0.5).add(0.5)
                fake = fake.mul(255)  # 数值范围为[0.0~1.0]-->[0~255]
                fake = fake.squeeze().permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                input_tensor = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
                filename_i = './save_image/%s/lecam_image/%s/' % (opt.type, str(epoch)) + str(p) + '.jpg'
                cv2.imwrite(filename_i, input_tensor)

            torch.save(generator.state_dict(), './models/%s/lecam/gen_%d.pth' % (opt.type,epoch))
            torch.save(discriminator.state_dict(), './models/%s/lecam/dis_%d.pth' % (opt.type,epoch))

    fileObject = open('loss/%s/d_pre.txt' % opt.type, 'w')
    for _ in d_pre:
        fileObject.write(str(_))
        fileObject.write(',')
    fileObject.close()

    fileObject = open('loss/%s/g_pre.txt' % opt.type, 'w')
    for _ in g_pre:
        fileObject.write(str(_))
        fileObject.write(',')
    fileObject.close()

    fileObject = open('loss/%s/d_los.txt' % opt.type, 'w')
    for _ in d_los:
        fileObject.write(str(_))
        fileObject.write(',')
    fileObject.close()

    fileObject = open('loss/%s/g_los.txt' % opt.type, 'w')
    for _ in g_los:
        fileObject.write(str(_))
        fileObject.write(',')
    fileObject.close()


def test():
    generator.load_state_dict(torch.load('models/ce/lecam/gen_450.pth'))    # 选择自己想使用的模型  gen_数字.pth
    generator.eval()
    for p in range(100):
        noise = Variable(Tensor(np.random.normal(0, 1, (1, 128))))
        fake = generator(noise)
        fake = fake.mul(0.5).add(0.5)
        fake = fake.mul(255)  # 数值范围为[0.0~1.0]-->[0~255]
        fake = fake.squeeze().permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        fake = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)

        filename_i = './save_image/agg/lecam_image/gen_image/%d' % p + '.jpg'
        cv2.imwrite(filename_i, fake)


if __name__ == '__main__':

    train()
    # test()