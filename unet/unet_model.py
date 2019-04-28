# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(513, 513)
        self.up1 = up(1026, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        probs = x.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("исходное")
        plt.imshow(mask)
        plt.savefig('file.png')
        plt.show()


        x1 = self.inc(x)
        print("x1 - ", x1.shape)
        probs = x1.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("inc")
        plt.imshow(mask)
        plt.savefig('inc.png')
        plt.show()


        x2 = self.down1(x1)
        print("x2 - ", x2.shape)
        probs = x2.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("down1")
        plt.imshow(mask)
        plt.savefig('down1.png')
        plt.show()


        x3 = self.down2(x2)
        print("x3 - ", x3.shape)
        probs = x3.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("down2")
        plt.imshow(mask)
        plt.savefig('down2.png')
        plt.show()

        x4 = self.down3(x3)
        print("x4 - ", x4.shape)
        probs = x4.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("down3")
        plt.imshow(mask)
        plt.savefig('down3.png')
        plt.show()

        mapp = np.load("56.npy")
        op = [[mapp]]
        x4 = torch.cat([x4, torch.Tensor(op)], dim=1)

        x5 = self.down4(x4)
        print("x5 - ", x5.shape)
        probs = x5.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("down4")
        plt.imshow(mask)
        plt.savefig('down4.png')
        plt.show()

        x = self.up1(x5, x4)
        print("up1 - ", x.shape)
        probs = x.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("up1")
        plt.imshow(mask)
        plt.savefig('up1.png')
        plt.show()


        x = self.up2(x, x3)
        print("up2 - ", x.shape)
        probs = x.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("up2")
        plt.imshow(mask)
        plt.savefig('up2.png')
        plt.show()


        x = self.up3(x, x2)
        print("up3 - ", x.shape)
        probs = x.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("up3")
        plt.imshow(mask)
        plt.savefig('up3.png')
        plt.show()


        x = self.up4(x, x1)
        print("up4 - ", x.shape)
        probs = x.squeeze(0)
        probs = probs[1, :, :]
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("up4")
        plt.imshow(mask)
        plt.savefig('up4.png')
        plt.show()


        x = self.outc(x)
        print("outc - ", x.shape)
        probs = x.squeeze(0)
        print(probs.shape)
        # probs = np.transpose(probs, axes=[2, 1, 0])
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        # mask = np.transpose(mask, axes=[1, 0, 2])
        print(mask.shape)
        plt.title("outc")
        plt.imshow(mask)
        plt.savefig('outc.png')
        plt.show()


        return F.sigmoid(x)
