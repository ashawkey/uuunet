# this is where magic happens.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import *
from uuunet.network_architecture.neural_network import SegmentationNetwork
from time import time
import numpy as np


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: [B, C, H, W, D]
        s = F.adaptive_avg_pool2d(x, 1) # [B, C, 1, 1, 1]
        s = self.conv1(s) # [B, C//reduction, 1, 1, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s) # [B, C, 1, 1, 1]
        x = x + torch.sigmoid(s)
        return x

class ConvBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1, is_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_activation = is_activation
        
        if is_activation:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x


class SENextBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None, groups=32, reduction=16, pool=None, is_shortcut=False, dilation=1):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBR2d(in_channels, mid_channels, 1, 0, 1)
        self.conv2 = ConvBR2d(mid_channels, mid_channels, 3, padding=dilation, stride=1, dilation=dilation, groups=groups)
        self.conv3 = ConvBR2d(mid_channels, out_channels, 1, 0, 1, is_activation=False)
        self.se = SEModule(out_channels, reduction)
        self.stride = stride
        self.is_shortcut = is_shortcut
        self.pool = pool
        
        if is_shortcut:
            self.shortcut = ConvBR2d(in_channels, out_channels, 1, 0, 1, is_activation=False)
    
    def forward(self, x):
        s = self.conv1(x)
        s = self.conv2(s)
        if self.stride is not None:
            if self.pool == 'max':
                s = F.max_pool2d(s, self.stride, self.stride)
            elif self.pool == 'avg':
                s = F.avg_pool2d(s, self.stride, self.stride)
        s = self.conv3(s)
        s = self.se(s)
        
        if self.is_shortcut: # #channels changed
            if self.stride is not None:
                x = F.avg_pool2d(x, self.stride, self.stride) # avg
            x = self.shortcut(x)
        
        x = x + s
        x = F.relu(x, inplace=True)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = ConvBR2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBR2d(out_channels, out_channels, kernel_size=3, padding=1)
        # att
        #self.att = Attention(in_channels, skip_channels, skip_channels//2)
        
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=self.stride, mode="bilinear")
        #skip = self.att(x, skip) # gate skip with x
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class PAM(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class XNet(SegmentationNetwork):
    def __init__(self, 
                 num_features=1,
                 ):
        super().__init__()
        
        backbone = [2,2,2,2]
        encoder_channels = [64, 128, 192, 256, 512]
        decoder_channels = [256, 128, 64, 32]
        strides = [[2,2], [2,2], [2,2], [2,2]]

        self.conv_op = nn.Conv2d
        self.input_shape_must_be_divisible_by = np.prod(strides, 0, dtype=np.int64)

        ### MR encoder [256, 256]
        self.block00 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block01 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block02 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )
        self.block03 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=None, is_shortcut=True, dilation=2),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False, dilation=2) for i in range(backbone[2])]
        )
        self.block04 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=None, is_shortcut=True, dilation=4),
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False, dilation=4) for i in range(backbone[3])]
        )  

        ### MR decoder
        #self.deblock04 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])
        #self.deblock03 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        #self.deblock02 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        #self.deblock01 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])


        ### one more 
        backbone = [2,2,2,2,2]
        encoder_channels = [32, 64, 128, 192, 256, 512]
        decoder_channels = [256, 128, 64, 32, 16]
        strides = [[2,2], [2,2], [2,2], [2,2], [2,2]]

        ### CT encoder [512, 512]
        self.block10 = nn.Sequential(
            ConvBR2d(num_features, encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
            ConvBR2d(encoder_channels[0], encoder_channels[0], kernel_size=3, stride=1, padding=1),
        )
        self.block11 = nn.Sequential(
            SENextBottleneck(encoder_channels[0], encoder_channels[1], stride=strides[0], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[1], encoder_channels[1], stride=None, is_shortcut=False) for i in range(backbone[0])]
        )
        self.block12 = nn.Sequential(
            SENextBottleneck(encoder_channels[1], encoder_channels[2], stride=strides[1], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[2], encoder_channels[2], stride=None, is_shortcut=False) for i in range(backbone[1])]
        )
        self.block13 = nn.Sequential(
            SENextBottleneck(encoder_channels[2], encoder_channels[3], stride=strides[2], is_shortcut=True, pool='max'),
          *[SENextBottleneck(encoder_channels[3], encoder_channels[3], stride=None, is_shortcut=False) for i in range(backbone[2])]
        )
        self.block14 = nn.Sequential(
            SENextBottleneck(encoder_channels[3], encoder_channels[4], stride=None, is_shortcut=True, dilation=2),
          *[SENextBottleneck(encoder_channels[4], encoder_channels[4], stride=None, is_shortcut=False, dilation=2) for i in range(backbone[3])]
        )
        self.block15 = nn.Sequential(
            SENextBottleneck(encoder_channels[4], encoder_channels[5], stride=None, is_shortcut=True, dilation=4),
          *[SENextBottleneck(encoder_channels[5], encoder_channels[5], stride=None, is_shortcut=False, dilation=4) for i in range(backbone[4])]
        )
        

        ### CT decoder
        #self.deblock15 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], decoder_channels[0], strides[-1])
        #self.deblock14 = DecoderBlock(decoder_channels[0], encoder_channels[-3], decoder_channels[1], strides[-2])
        #self.deblock13 = DecoderBlock(decoder_channels[1], encoder_channels[-4], decoder_channels[2], strides[-3])
        #self.deblock12 = DecoderBlock(decoder_channels[2], encoder_channels[-5], decoder_channels[3], strides[-4])
        #self.deblock11 = DecoderBlock(decoder_channels[3], encoder_channels[-6], decoder_channels[4], strides[-5])

        ### Shared decoder
        self.decoder = nn.Sequential(
            ConvBR2d(encoder_channels[-1], encoder_channels[-1], 3, 1, 1),
            PAM(encoder_channels[-1]),
            ConvBR2d(encoder_channels[-1], encoder_channels[-1], 3, 1, 1),
        )

        ### MR seg head
        self.head = nn.Conv2d(encoder_channels[-1], 5, 1, 1, 0, bias=False)

        ### CT seg head
        #self.head1 = nn.Conv2d(decoder_channels[-1], 2, 1, 1, 0, bias=False)

        self.cnt = 0

    def forward(self, x, modality):
        size = x.shape[2:]
        if modality == 0:
            _x_ = x.detach().cpu().numpy() # [B, 1, 256, 256]
            x0 = self.block00(x)
            x1 = self.block01(x0)
            x2 = self.block02(x1)
            x3 = self.block03(x2)
            x4 = self.block04(x3)
            x = self.decoder(x4)
            _f_ = x.detach().cpu().numpy() # [B, 512, 32, 32]
            x = self.head(x)
            _y_ = x.detach().cpu().numpy().argmax(1) # [B, 1, 256, 256]
            print(self.cnt, _x_.shape, _f_.shape, _y_.shape)
            torch.save({"x":_x_, "f":_f_, "y":_y_}, f'{modality}_{self.cnt}.bin')
            self.cnt += 1
        else:
            _x_ = x.detach().cpu().numpy() # [B, 1, 256, 256]
            x0 = self.block10(x)
            x1 = self.block11(x0)
            x2 = self.block12(x1)
            x3 = self.block13(x2)
            x4 = self.block14(x3)
            x5 = self.block15(x4)
            x = self.decoder(x5)
            _f_ = x.detach().cpu().numpy() # [B, 512, 32, 32]
            x = self.head(x)            
            _y_ = x.detach().cpu().numpy().argmax(1) # [B, 1, 256, 256]
            print(self.cnt, _x_.shape, _f_.shape, _y_.shape)
            torch.save({"x":_x_, "f":_f_, "y":_y_}, f'{modality}_{self.cnt}.bin')
            self.cnt += 1
        x = F.interpolate(x, size=size, mode='bilinear')
        return x


# single trainer
# plan A: train MR fully, then train CT fully
def run_training_A(trainer, modality):
    torch.cuda.empty_cache()

    trainer._maybe_init_amp()

    maybe_mkdir_p(trainer.output_folder)

    if not trainer.was_initialized:
        trainer.initialize(True)

    while trainer.epoch < trainer.max_num_epochs:
        trainer.print_to_log_file("\nepoch: ", trainer.epoch)
        epoch_start_time = time()
        train_losses_epoch = []

        ####################
        #  train one epoch
        trainer.network.train()
        for b in range(trainer.num_batches_per_epoch):
            l = trainer.run_iteration(trainer.tr_gen, modality, True)
            train_losses_epoch.append(l)
        ####################

        trainer.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer.print_to_log_file("train loss : %.4f" % trainer.all_tr_losses[-1])

        with torch.no_grad():
            # validation with train=False
            trainer.network.eval()
            val_losses = []

            ###############################
            # validate one epoch
            for b in range(trainer.num_val_batches_per_epoch):
                l = trainer.run_iteration(trainer.val_gen, modality, False, True)
                val_losses.append(l)
            ###############################

            trainer.all_val_losses.append(np.mean(val_losses))
            trainer.print_to_log_file("val loss (train=False): %.4f" % trainer.all_val_losses[-1])

        epoch_end_time = time()

        trainer.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        continue_training = trainer.on_epoch_end()
        
        # logging to tensorboard
        trainer.writer.add_scalar(f"train/loss{modality}", trainer.all_tr_losses[-1], trainer.epoch)
        trainer.writer.add_scalar(f"evaulate/loss{modality}", trainer.all_val_losses[-1], trainer.epoch)
        trainer.writer.add_scalar(f"evaluate/metric{modality}", trainer.all_val_eval_metrics[-1], trainer.epoch)

        if not continue_training:
            # allows for early stopping
            break

        trainer.epoch += 1
        trainer.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

    trainer.save_checkpoint(join(trainer.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer.output_folder, "model_latest.model")):
        os.remove(join(trainer.output_folder, "model_latest.model"))
    if isfile(join(trainer.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer.output_folder, "model_latest.model.pkl"))

    
# two trainers
# plan B: epoch-wise cross 
def run_training_B(trainer0, trainer1):
    torch.cuda.empty_cache()

    trainer0._maybe_init_amp()
    trainer1._maybe_init_amp()

    maybe_mkdir_p(trainer0.output_folder)
    maybe_mkdir_p(trainer1.output_folder)

    if not trainer0.was_initialized:
        trainer0.initialize(True)

    if not trainer1.was_initialized:
        trainer1.initialize(True)
    
    while trainer0.epoch < trainer0.max_num_epochs and trainer1.epoch < trainer1.max_num_epochs:
        
        epoch_start_time = time()

        ####################
        #  trainer0 one epoch
        trainer0.print_to_log_file("\n[Trainer0] epoch: ", trainer0.epoch)
        train_losses_epoch = []
        trainer0.network.train()
        for b in range(trainer0.num_batches_per_epoch):
            l = trainer0.run_iteration(trainer0.tr_gen, 0, True)
            train_losses_epoch.append(l)
        trainer0.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer0.print_to_log_file("[Trainer0] train loss : %.4f" % trainer0.all_tr_losses[-1])
        ####################

        ####################
        #  trainer1 one epoch
        trainer1.print_to_log_file("\n[Trainer1] epoch: ", trainer1.epoch)
        train_losses_epoch = []
        trainer1.network.train()
        for b in range(trainer1.num_batches_per_epoch):
            l = trainer1.run_iteration(trainer1.tr_gen, 1, True)
            train_losses_epoch.append(l)
        trainer1.all_tr_losses.append(np.mean(train_losses_epoch))
        trainer1.print_to_log_file("[Trainer1] train loss : %.4f" % trainer1.all_tr_losses[-1])
        ####################


        with torch.no_grad():

            ###############################
            # validate0 one epoch
            trainer0.network.eval()
            val_losses = []
            for b in range(trainer0.num_val_batches_per_epoch):
                l = trainer0.run_iteration(trainer0.val_gen, 0, False, True)
                val_losses.append(l)
            trainer0.all_val_losses.append(np.mean(val_losses))
            trainer0.print_to_log_file("[Trainer0] val loss: %.4f" % trainer0.all_val_losses[-1])
            ###############################

            ###############################
            # validate1 one epoch
            trainer1.network.eval()
            val_losses = []
            for b in range(trainer1.num_val_batches_per_epoch):
                l = trainer1.run_iteration(trainer1.val_gen, 1, False, True)
                val_losses.append(l)
            trainer1.all_val_losses.append(np.mean(val_losses))
            trainer1.print_to_log_file("[Trainer1] val loss: %.4f" % trainer1.all_val_losses[-1])
            ###############################

        epoch_end_time = time()

        trainer0.update_train_loss_MA()  # needed for lr scheduler and stopping of training
        trainer1.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        # this may cause saving two same models, but anyway it should work.
        continue_training = trainer0.on_epoch_end()
        continue_training = trainer1.on_epoch_end()

        # logging to tensorboard
        trainer0.writer.add_scalar("train/loss0", trainer0.all_tr_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaulate/loss0", trainer0.all_val_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaluate/metric0", trainer0.all_val_eval_metrics[-1], trainer0.epoch)
        # logging to tensorboard
        trainer1.writer.add_scalar("train/loss1", trainer1.all_tr_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaulate/loss1", trainer1.all_val_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaluate/metric1", trainer1.all_val_eval_metrics[-1], trainer1.epoch)

        trainer0.epoch += 1
        trainer1.epoch += 1

        # also, those two log files are differently placed.
        trainer0.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))
        trainer1.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))

    # save once is OK
    trainer0.save_checkpoint(join(trainer0.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer0.output_folder, "model_latest.model")):
        os.remove(join(trainer0.output_folder, "model_latest.model"))
    if isfile(join(trainer0.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer0.output_folder, "model_latest.model.pkl"))


# two trainers
# plan C: step-wise cross 
def run_training_C(trainer0, trainer1):
    torch.cuda.empty_cache()

    trainer0._maybe_init_amp()
    trainer1._maybe_init_amp()

    maybe_mkdir_p(trainer0.output_folder)
    maybe_mkdir_p(trainer1.output_folder)

    if not trainer0.was_initialized:
        trainer0.initialize(True)

    if not trainer1.was_initialized:
        trainer1.initialize(True)
    
    while trainer0.epoch < trainer0.max_num_epochs and trainer1.epoch < trainer1.max_num_epochs:

        epoch_start_time = time()

        ####################
        #  trainer0 & trainer1 one epoch
        trainer0.print_to_log_file("\n[Trainer0] epoch: ", trainer0.epoch)
        trainer1.print_to_log_file("\n[Trainer1] epoch: ", trainer1.epoch)
        
        train_losses_epoch0 = []
        train_losses_epoch1 = []

        trainer0.network.train()
        trainer1.network.train()

        for b in range(trainer0.num_batches_per_epoch*2): # double steps
            if b%2 == 0:
                # trainer0
                l = trainer0.run_iteration(trainer0.tr_gen, 0, True)
                train_losses_epoch0.append(l)
            else:
                # trainer1
                l = trainer1.run_iteration(trainer1.tr_gen, 1, True)
                train_losses_epoch1.append(l)

        trainer0.all_tr_losses.append(np.mean(train_losses_epoch0))
        trainer1.all_tr_losses.append(np.mean(train_losses_epoch1))
        trainer0.print_to_log_file("[Trainer0] train loss : %.4f" % trainer0.all_tr_losses[-1])
        trainer1.print_to_log_file("[Trainer1] train loss : %.4f" % trainer1.all_tr_losses[-1])
        ####################


        with torch.no_grad():

            ###############################
            # validate0 one epoch
            trainer0.network.eval()
            val_losses = []
            for b in range(trainer0.num_val_batches_per_epoch):
                l = trainer0.run_iteration(trainer0.val_gen, 0, False, True)
                val_losses.append(l)
            trainer0.all_val_losses.append(np.mean(val_losses))
            trainer0.print_to_log_file("[Trainer0] val loss: %.4f" % trainer0.all_val_losses[-1])
            ###############################

            ###############################
            # validate1 one epoch
            trainer1.network.eval()
            val_losses = []
            for b in range(trainer1.num_val_batches_per_epoch):
                l = trainer1.run_iteration(trainer1.val_gen, 1, False, True)
                val_losses.append(l)
            trainer1.all_val_losses.append(np.mean(val_losses))
            trainer1.print_to_log_file("[Trainer1] val loss: %.4f" % trainer1.all_val_losses[-1])
            ###############################

        epoch_end_time = time()

        trainer0.update_train_loss_MA()  # needed for lr scheduler and stopping of training
        trainer1.update_train_loss_MA()  # needed for lr scheduler and stopping of training

        # this may cause saving two same models, but anyway it should work.
        continue_training = trainer0.on_epoch_end()
        continue_training = trainer1.on_epoch_end()

        # logging to tensorboard
        trainer0.writer.add_scalar("train/loss0", trainer0.all_tr_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaulate/loss0", trainer0.all_val_losses[-1], trainer0.epoch)
        trainer0.writer.add_scalar("evaluate/metric0", trainer0.all_val_eval_metrics[-1], trainer0.epoch)
        # logging to tensorboard
        trainer1.writer.add_scalar("train/loss1", trainer1.all_tr_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaulate/loss1", trainer1.all_val_losses[-1], trainer1.epoch)
        trainer1.writer.add_scalar("evaluate/metric1", trainer1.all_val_eval_metrics[-1], trainer1.epoch)

        trainer0.epoch += 1
        trainer1.epoch += 1

        # also, those two log files are differently placed.
        trainer0.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))
        trainer1.print_to_log_file("This epoch took %f s\n" % (epoch_end_time-epoch_start_time))


    trainer0.save_checkpoint(join(trainer0.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer0.output_folder, "model_latest.model")):
        os.remove(join(trainer0.output_folder, "model_latest.model"))
    if isfile(join(trainer0.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer0.output_folder, "model_latest.model.pkl"))

    trainer1.save_checkpoint(join(trainer1.output_folder, "model_final_checkpoint.model"))
    # now we can delete latest as it will be identical with final
    if isfile(join(trainer1.output_folder, "model_latest.model")):
        os.remove(join(trainer1.output_folder, "model_latest.model"))
    if isfile(join(trainer1.output_folder, "model_latest.model.pkl")):
        os.remove(join(trainer1.output_folder, "model_latest.model.pkl"))