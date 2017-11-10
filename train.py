import math
import os

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from misc import PerceptualLoss, AverageMeter
from models import Generator, Discriminator

train_args = {
    'train_batch_size': 256,
    'val_batch_size': 1024,
    'hr_size': 96,  # make sure that hr_size can be divided by scale_factor exactly
    'scale_factor': 4,  # should be power of 2
    'g_snapshot': '',
    'd_snapshot': '',
    'g_lr': 1e-4,
    'd_lr': 1e-4,
    'dataset_root': '/media/b3-542/340E7D380E7CF3E8/places365_standard',
    'start_epoch': 1,
    'epoch_num': 14,
    'ckpt_path': './ckpt'
}

g_pretrain_args = {
    'pretrain': True,
    'epoch_num': 140,
    'lr': 1e-4,
}

writer = SummaryWriter(train_args['ckpt_path'])

hr_transform = transforms.Compose([
    transforms.RandomCrop(train_args['hr_size']),
    transforms.ToTensor()
])
lr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(train_args['hr_size'] / train_args['scale_factor'], interpolation=3),
    transforms.ToTensor()
])
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denormalize = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
])
lr_bicubic_upscale = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(train_args['hr_size'], interpolation=3),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder(os.path.join(train_args['dataset_root'], 'train'), hr_transform)
train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], shuffle=True, num_workers=8,
                          pin_memory=True)
val_set = datasets.ImageFolder(os.path.join(train_args['dataset_root'], 'val'), hr_transform)
val_loader = DataLoader(val_set, batch_size=train_args['val_batch_size'], num_workers=8, pin_memory=True)

g = Generator(scale_factor=train_args['scale_factor']).cuda().train()
if len(train_args['g_snapshot']) > 0:
    g.load_state_dict(train_args['g_snapshot'])

mse_criterion = nn.MSELoss().cuda()
train_iter_num = len(train_loader)
val_iter_num = len(val_loader)

g_mse_loss_record, g_perceptual_loss_record, g_ad_loss_record = AverageMeter(), AverageMeter(), AverageMeter()
g_loss_record, d_loss_record, psnr_record = AverageMeter(), AverageMeter(), AverageMeter()

if g_pretrain_args['pretrain']:
    g_optimizer = optim.Adam(g.parameters(), lr=g_pretrain_args['lr'])
    for epoch in range(g_pretrain_args['epoch_num']):
        for i, data in enumerate(train_loader):
            hr_imgs, _ = data
            batch_size = hr_imgs.size(0)
            lr_imgs = Variable(torch.stack([normalize(lr_transform(img)) for img in hr_imgs], 0)).cuda()
            hr_imgs = Variable(torch.stack([normalize(img) for img in hr_imgs], 0)).cuda()

            g.zero_grad()
            gen_hr_imgs = g(lr_imgs)
            loss = mse_criterion(gen_hr_imgs, hr_imgs)
            loss.backward()
            g_optimizer.step()

            g_mse_loss_record.update(loss.data[0], batch_size)
            psnr_record.update(10 * math.log10(1 / loss.data[0]), batch_size)

            print '[pretrain]: [epoch %d], [iter %d / %d], [loss %.4f], [psnr %.4f]' % (
                epoch + 1, i + 1, train_iter_num, g_mse_loss_record.avg, psnr_record.avg)

            writer.add_scalar('pretrain_g_mse_loss', g_mse_loss_record.avg, epoch * train_iter_num + i + 1)
            writer.add_scalar('pretrain_psnr', psnr_record.avg, epoch * train_iter_num + i + 1)

        torch.save(g.state_dict(), 'pretrain_g_epoch_%d_loss_%.4f_psnr_%.4f.pth' % (
            epoch + 1, g_mse_loss_record.avg, psnr_record.avg))

        g_mse_loss_record.reset()
        psnr_record.reset()

d = Discriminator().cuda().train()
if len(train_args['d_snapshot']) > 0:
    d.load_state_dict(train_args['d_snapshot'])
perceptual_criterion = PerceptualLoss().cuda()
ad_criterion = nn.BCELoss().cuda()
g_optimizer = optim.Adam(g.parameters(), lr=train_args['g_lr'])
d_optimizer = optim.Adam(d.parameters(), lr=train_args['d_lr'])

for epoch in range(train_args['start_epoch'] - 1, train_args['epoch_num']):
    for i, data in enumerate(train_loader):
        hr_imgs, _ = data
        batch_size = hr_imgs.size(0)
        lr_imgs = Variable(torch.stack([normalize(lr_transform(img)) for img in hr_imgs], 0)).cuda()
        hr_imgs = Variable(torch.stack([normalize(img) for img in hr_imgs], 0)).cuda()
        gen_hr_imgs = g(lr_imgs)
        target_real = Variable(torch.ones(batch_size)).cuda()
        target_fake = Variable(torch.zeros(batch_size)).cuda()

        # update d
        d.zero_grad()
        d_ad_loss = ad_criterion(d(hr_imgs), target_real) + ad_criterion(d(gen_hr_imgs.detach()), target_fake)
        d_ad_loss.backward()
        d_optimizer.step()

        d_loss_record.update(d_ad_loss.data[0], batch_size)

        # update g
        g.zero_grad()
        g_mse_loss = mse_criterion(gen_hr_imgs, hr_imgs)
        g_perceptual_loss = perceptual_criterion(gen_hr_imgs, hr_imgs)
        g_ad_loss = ad_criterion(d(gen_hr_imgs), target_real)
        g_loss = g_mse_loss + 0.006 * g_perceptual_loss + 0.001 * g_ad_loss
        g_loss.backward()
        g_optimizer.step()

        g_mse_loss_record.update(g_mse_loss.data[0], batch_size)
        g_perceptual_loss_record.update(g_perceptual_loss.data[0], batch_size)
        g_ad_loss_record.update(g_ad_loss.data[0], batch_size)
        g_loss_record.update(g_loss.data[0], batch_size)
        psnr_record.update(10 * math.log10(1 / g_mse_loss.data[0]), batch_size)

        print '[train]: [epoch %d], [iter %d / %d], [d_ad_loss %.4f], [g_ad_loss %.4f], [psnr %.4f],' \
              '[g_mse_loss %.4f], [g_perceptual_loss %.4f], [g_loss %.4f]' % \
              (epoch + 1, i + 1, train_iter_num, d_loss_record.avg, g_ad_loss_record.avg, psnr_record.avg,
               g_mse_loss_record.avg, g_perceptual_loss_record.avg, g_loss_record.avg)

        writer.add_scalar('d_loss', d_loss_record.avg, epoch * train_iter_num + i + 1)
        writer.add_scalar('g_mse_loss', g_mse_loss_record.avg, epoch * train_iter_num + i + 1)
        writer.add_scalar('g_perceptual_loss', g_perceptual_loss_record.avg, epoch * train_iter_num + i + 1)
        writer.add_scalar('g_ad_loss', g_ad_loss_record.avg, epoch * train_iter_num + i + 1)
        writer.add_scalar('g_loss', g_loss_record.avg, epoch * train_iter_num + i + 1)
        writer.add_scalar('psnr', psnr_record.avg, epoch * train_iter_num + i + 1)

    d_loss_record.reset()
    g_mse_loss_record.reset()
    g_perceptual_loss_record.reset()
    g_ad_loss_record.reset()
    g_loss_record.reset()
    psnr_record.reset()

    g.eval()

    val_visual = []
    for i, data in enumerate(val_loader):
        hr_imgs, _ = data
        batch_size = hr_imgs.size(0)
        lr_imgs = Variable(torch.stack([normalize(lr_transform(img)) for img in hr_imgs], 0), volatile=True).cuda()
        hr_imgs = Variable(torch.stack([normalize(img) for img in hr_imgs], 0), volatile=True).cuda()
        gen_hr_imgs = g(lr_imgs)

        g_mse_loss = mse_criterion(gen_hr_imgs, hr_imgs)
        g_perceptual_loss = perceptual_criterion(gen_hr_imgs, hr_imgs)

        g_mse_loss_record.update(g_mse_loss.data[0], batch_size)
        g_perceptual_loss_record.update(g_perceptual_loss.data[0], batch_size)
        psnr_record.update(10 * math.log10(1 / g_mse_loss.data[0]), batch_size)

        for lr, hr, hr_gen in zip(lr_imgs.cpu(), hr_imgs.cpu().data, gen_hr_imgs.cpu().data):
            val_visual.extend([lr_bicubic_upscale(denormalize(lr)), denormalize(hr), denormalize(hr_gen)])

        print 'validating %d / %d' % (i + 1, val_iter_num)

    val_visual = torch.stack(val_visual, 0)
    val_visual = vutils.make_grid(val_visual, nrow=15, padding=5)

    snapshot_name = 'epoch_%d_psnr_%.4f_g_mse_loss_%.4f_g_perceptual_loss_%.4f' % (
        epoch + 1, psnr_record.avg, g_mse_loss_record.avg, g_perceptual_loss_record.avg)

    writer.add_scalar('validate_psnr', psnr_record.avg, epoch + 1)
    writer.add_scalar('validate_g_mse_loss', g_mse_loss_record.avg, epoch + 1)
    writer.add_scalar('validate_g_perceptual_loss', g_perceptual_loss_record.avg, epoch + 1)
    writer.add_image(snapshot_name, val_visual)

    print '[validate]: [epoch %d], [psnr %.4f], [g_mse_loss %.4f], [g_perceptual_loss %.4f]' % (
        epoch + 1, psnr_record.avg, g_mse_loss_record.avg, g_perceptual_loss_record.avg)

    torch.save(g.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_g.pth'))
    torch.save(d.state_dict(), os.path.join(train_args['ckpt_path'], snapshot_name + '_d.pth'))

    psnr_record.reset()
    g_mse_loss_record.reset()
    g_perceptual_loss_record.reset()

    g.train()
