import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from modules.RFRNet import RFRNet
from modules.RFRNetv2 import RFRNetv7, VGG16FeatureExtractor
from utils.io import load_ckpt, save_ckpt

# GOT_AMP = False
# try:
#     print("Checking for Apex AMP support...")
#     from apex import amp
#     GOT_AMP = True
#     print(" - [x] yes")
# except ImportError:
#     print(" - [!] no")


def gram_matrix(input_tensor):
    """
    Compute Gram matrix
    :param input_tensor: input tensor with shape
    (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y

    Ripped from: https://github.com/NVIDIA/partialconv/blob/master/models/loss.py#L17-L40
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t,
                         beta=0, alpha=1./(ch * h * w), out=None)

    # naive way to avoid underflow for mixed precision training
    # features = features / (ch * h)
    # gram = features.bmm(features_t) / w

    # for fp32 training, it is also safe to use the following:
    # gram = features.bmm(features_t) / (ch * h * w)

    return gram


class RFRNetModel():
    def __init__(self):
        self.loss_weights = {
            "tv": 0.1,
            "style": 120,
            "perceptual": 0.05,
            "valid": 1,
            "hole": 6
        }
        self.learning_rates = {
            "train": 2e-4,
            "finetune": 5e-5
        }
        self.save_freq = 5000

        self.G = None
        self.lossNet = None
        self.optm_G = None
        self.device = None

        self.iter = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0

    def initialize_model(self, path=None, train=True):
        # self.G = RFRNet()
        self.G = RFRNetv7()
        self.additional_data = None
        self.optm_G = optim.Adam(self.G.parameters(),
                                 lr=self.learning_rates["train"])
        if train:
            self.lossNet = VGG16FeatureExtractor()

        try:
            start_iter = load_ckpt(path,
                                   [('generator', self.G)],
                                   [('optimizer_G', self.optm_G)])
            self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0

        return self

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")

            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        return self

    def multi_gpu(self):
        print(f'Multi-GPU training with {torch.cuda.device_count()} GPUs')
        self.G = nn.DataParallel(self.G)
        if self.lossNet is not None:
            self.lossNet = nn.DataParallel(self.lossNet)
        return self

    def train(self, train_loader, save_path, finetune=False, iters=450000,
              fp16=False,
              multi_gpu=True):
        writer = SummaryWriter()

        self.G.train(finetune=finetune)
        # Overwrite optimizer with a lower lr
        if finetune:
            for g in self.optm_G.param_groups:
                g['lr'] = self.learning_rates["finetune"]

        # self.fp16 = fp16 and GOT_AMP
        self.fp16 = fp16
        if self.fp16:
            # self.G, self.optm_G = amp.initialize(self.G, self.optm_G,
            #                                      opt_level="O1")
            # if self.lossNet is not None:
            #     self.lossNet = amp.initialize(self.lossNet,
            #                                   opt_level="O1")
            print("Creating grad scaler...")
            self.grad_scaler = GradScaler()

        if multi_gpu:
            self.multi_gpu()

        print("Starting training from iteration: {:d}, finetuning: {}".format(
            self.iter, finetune))
        s_time = time.time()
        while self.iter < iters:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks

                self.forward(masked_images, masks, gt_images)
                self.update_parameters()

                for k, v in self.metrics["lossG"].items():
                    writer.add_scalar(f"lossG/{k}", v,
                                      global_step=self.iter)

                self.iter += 1

                if self.iter % 200 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %
                          (self.iter, self.l1_loss_val/50, int_time))

                    writer.add_images("real_A", self.real_A,
                                      global_step=self.iter)
                    writer.add_images("mask", self.mask,
                                      global_step=self.iter)
                    writer.add_images("real_B", self.real_B,
                                      global_step=self.iter)
                    writer.add_images("fake_B", torch.clamp(self.fake_B, 0, 1),
                                      global_step=self.iter)
                    writer.add_images("comp_B", torch.clamp(self.comp_B, 0, 1),
                                      global_step=self.iter)

                    if self.additional_data is not None:
                        for i in range(len(self.additional_data[0])):
                            writer.add_images(f"recur_im_{i}", torch.clamp(self.additional_data[0][i], 0, 1),
                                              self.iter)
                            writer.add_images(f"recur_m_{i}", self.additional_data[1][i],
                                              self.iter)

                    # Reset
                    s_time = time.time()
                    self.l1_loss_val = 0.0

                if self.iter % self.save_freq == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    save_ckpt('{:s}/g_{:d}{}.pth'.format(save_path, self.iter, "_finetune" if finetune else ""),
                              [('generator', self.G)],
                              [('optimizer_G', self.optm_G)],
                              self.iter)

        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}{}.pth'.format(save_path, "final", "_finetune" if finetune else ""),
                      [('generator', self.G)],
                      [('optimizer_G', self.optm_G)],
                      self.iter)

    def test(self, test_loader, result_save_path):
        self.G.eval()

        with torch.no_grad():
            count = 0
            for items in test_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                if masks.size(1) == 1:
                    masks = torch.cat([masks, ] * 3, dim=1)

                fake_B, mask = self.G(masked_images, masks)
                comp_B = fake_B * (1 - masks) + gt_images * masks

                if not os.path.exists('{:s}/results'.format(result_save_path)):
                    os.makedirs('{:s}/results'.format(result_save_path))

                for k in range(comp_B.size(0)):
                    count += 1
                    grid = make_grid(comp_B[k:k+1])
                    file_path = '{:s}/results/img_{:d}.png'.format(result_save_path,
                                                                   count)
                    save_image(grid, file_path)

                    grid = make_grid(masked_images[k:k+1] + 1 - masks[k:k+1])
                    file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path,
                                                                          count)
                    save_image(grid, file_path)

    def forward(self, masked_image, mask, gt_image):
        with autocast(self.fp16):
            self.real_A = masked_image
            self.real_B = gt_image
            self.mask = mask

            # model's internal threads won't autocast.  The main thread's autocast state has no effect.
            # Ref: https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
            fake_B, additional_data = self.G(masked_image, mask,
                                             fp16=self.fp16)
            if additional_data is not None:
                self.additional_data = additional_data
            # assert not torch.isnan(fake_B).any()

            self.fake_B = fake_B
            self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask

    def update_parameters(self):
        self.update_G()
        self.update_D()

    def update_G(self):
        self.optm_G.zero_grad()

        with autocast(self.fp16):
            loss_G = self.get_g_loss()
        # if self.fp16:
        #     with amp.scale_loss(loss_G, self.optm_G) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss_G.backward()

        if self.fp16:
            self.grad_scaler.scale(loss_G).backward()
            self.grad_scaler.step(self.optm_G)
            self.grad_scaler.update()
        else:
            loss_G.backward()
            self.optm_G.step()

    def update_D(self):
        return

    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B

        real_B_feats = self.lossNet(real_B, fp16=self.fp16)
        fake_B_feats = self.lossNet(fake_B, fp16=self.fp16)
        comp_B_feats = self.lossNet(comp_B, fp16=self.fp16)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) \
            + self.style_loss(real_B_feats, comp_B_feats)
        perceptual_loss = self.perceptual_loss(real_B_feats, fake_B_feats) \
            + self.perceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)

        if self.additional_data is not None:
            for i, (im, m) in enumerate(zip(*self.additional_data)):
                valid_loss += self.l1_loss(real_B, im, m)

        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))

        loss_G = (tv_loss * self.loss_weights["tv"]
                  + style_loss * self.loss_weights["style"]
                  + perceptual_loss * self.loss_weights["perceptual"]
                  + valid_loss * self.loss_weights["valid"]
                  + hole_loss * self.loss_weights["hole"])

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()

        self.metrics = {
            "lossG": {
                "sum": loss_G.item(),
                "tv": tv_loss.item() * self.loss_weights["tv"],
                "style": style_loss.item() * self.loss_weights["style"],
                "perceptual": perceptual_loss.item() * self.loss_weights["perceptual"],
                "valid": valid_loss.item() * self.loss_weights["valid"],
                "hole": hole_loss.item() * self.loss_weights["hole"],
            }
        }
        print(f"#{self.iter:08d} - lossG: {self.metrics['lossG']}")

        return loss_G

    @staticmethod
    def l1_loss(f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    @staticmethod
    def style_loss(A_feats, B_feats):
        assert len(A_feats) == len(B_feats), \
            "the length of two input feature maps lists should be the same"

        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]

            # _, c, w, h = A_feat.size()
            # A_feat = A_feat.view(A_feat.size(0),
            #                      A_feat.size(1),
            #                      A_feat.size(2) * A_feat.size(3))
            # B_feat = B_feat.view(B_feat.size(0),
            #                      B_feat.size(1),
            #                      B_feat.size(2) * B_feat.size(3))
            # A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            # B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            # loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))

            # Avoid underflow when using mixed precision training
            gram_A = gram_matrix(A_feat)
            gram_B = gram_matrix(B_feat)
            loss_value += torch.mean(torch.abs(gram_A - gram_B))

        return loss_value

    @staticmethod
    def TV_loss(x):
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv

    @staticmethod
    def perceptual_loss(A_feats, B_feats):
        assert len(A_feats) == len(B_feats), \
            "the length of two input feature maps lists should be the same"

        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))

        return loss_value

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)


if __name__ == "__main__":
    x = torch.rand(2, 64, 256, 256)
    x_clone = x.clone()

    _, c, w, h = x.size()
    x = x.view(x.size(0),
               x.size(1),
               x.size(2) * x.size(3))
    gram_1 = torch.matmul(x, x.transpose(2, 1))/(c * w * h)
    gram_2 = gram_matrix(x_clone)

    assert torch.allclose(gram_1, gram_2), "gram_1 != gram_2"
