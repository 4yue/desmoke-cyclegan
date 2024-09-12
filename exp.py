import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
import itertools

from tqdm import tqdm
from data.dataloader import load_data
from utils import *
import models.networks as networks
from models.discriminators import Discriminator
from models.generator import SmokeRemover, SmokeAdder


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()

        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        # self.loss_function = self.args.loss

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

        if self.args.resume_path is not None:
            self._load_model()
    def _load_model(self):
        self.model.load_state_dict(torch.load(self.args.resume_path))

    def _build_model(self):
        """
        smoke_remover
        smoke_adder
        discriminator_smoke
        discriminator_clean
        """
        self.smoke_remover = SmokeRemover(self.args.input_nc, self.args.output_nc)
        self.smoke_adder = SmokeAdder(self.args.output_nc, self.args.input_nc)
        if self.args.is_train:  # define discriminators
            self.netD_smoke = Discriminator(self.args.input_nc).to(self.device)
            self.netD_clean = Discriminator(self.args.output_nc).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader = load_data(**config)

    def _select_optimizer(self):
        self.optimizer_G_Remover = torch.optim.Adam(self.smoke_remover.parameters(), lr=self.args.lr)
        self.optimizer_G_Adder = torch.optim.Adam(self.smoke_adder.parameters(), lr=self.args.lr)
        if self.args.is_train:
            self.optimizer_D_smoke = torch.optim.Adam(self.netD_smoke.parameters(), lr=self.args.lr)
            self.optimizer_D_clean = torch.optim.Adam(self.netD_clean.parameters(), lr=self.args.lr)

    def _select_criterion(self):
        # todo
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        self.criterionPred = torch.nn.L1Loss()

        self.loss_names = ['Gan_R', 'Gan_A', 'Cycle_R', 'Cycle_A', 'Idt_R', 'Idt_A', 'Pred', 'G', 'D_smoke', 'D_clean']

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__

        for epoch in range(config['epochs']):
            self.smoke_remover.train()
            self.smoke_adder.train()
            self.netD_smoke.train()
            self.netD_clean.train()
            train_pbar = tqdm(self.train_loader)

            Gan_R, Gan_A, Cycle_R, Cycle_A, Idt_R, Idt_A, Pred, G, D_smoke, D_clean = [], [], [], [], [], [], [], [], [], []
            real_smoke_lst, real_clean_lst, pred_smoke_lst, pred_clean_lst = [], [], [], []
            fake_smoke_lst, fake_clean_lst, rec_smoke_lst, rec_clean_lst = [], [], [], []
            for real_smoke, real_clean, pred_smoke, pred_clean in train_pbar:

                # forward
                fake_clean = self.smoke_remover(real_smoke, pred_smoke)
                rec_smoke = self.smoke_adder(fake_clean)
                fake_smoke = self.smoke_adder(real_clean)
                rec_clean = self.smoke_remover(fake_smoke, pred_clean)

                list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()),
                         [real_smoke, real_clean, pred_smoke, pred_clean, fake_clean, rec_smoke, fake_smoke, rec_clean],
                         [real_smoke_lst, real_clean_lst, pred_smoke_lst, pred_clean_lst, fake_clean_lst, rec_smoke_lst, fake_smoke_lst, rec_clean_lst]))

                # backward_G
                self.optimizer_G_Remover.zero_grad()
                self.optimizer_G_Adder.zero_grad()
                # GAN loss D_clean(remover(real_smoke, pred_smoke))
                loss_GAN_remover = self.criterionGAN(self.netD_clean(fake_clean), True)
                loss_GAN_adder = self.criterionGAN(self.netD_smoke(fake_smoke), True)

                # cycle loss
                loss_cycle_remover = self.criterionCycle(rec_smoke, real_smoke)
                loss_cycle_adder = self.criterionCycle(rec_clean, real_clean)

                # todo identity loss
                # loss_identity_remover = 0
                # loss_identity_adder = 0

                # todo prediction loss
                # loss_pred = 0

                # loss_G
                # loss_G = loss_GAN_remover + loss_GAN_adder + loss_cycle_remover + loss_cycle_adder + loss_identity_remover + loss_identity_adder + loss_pred
                loss_G = loss_GAN_remover + loss_GAN_adder + loss_cycle_remover + loss_cycle_adder
                loss_G.backward()
                self.optimizer_G_Remover.step()
                self.optimizer_G_Adder.step()

                # backward_D_smoke
                self.optimizer_D_smoke.zero_grad()
                loss_D_smoke_real = self.criterionGAN(self.netD_smoke(real_smoke), True)
                loss_D_smoke_fake = self.criterionGAN(self.netD_smoke(fake_smoke), False)
                loss_D_smoke = (loss_D_smoke_real + loss_D_smoke_fake)
                loss_D_smoke.backward()
                self.optimizer_D_smoke.step()

                # backward_D_clean
                self.optimizer_D_clean.zero_grad()
                loss_D_clean_real = self.criterionGAN(self.netD_clean(real_clean), True)
                loss_D_clean_fake = self.criterionGAN(self.netD_clean(fake_clean), False)
                loss_D_clean = (loss_D_clean_real + loss_D_clean_fake)
                loss_D_clean.backward()
                self.optimizer_D_clean.step()

                Gan_R.append(loss_GAN_remover.item())
                Gan_A.append(loss_GAN_adder.item())
                Cycle_R.append(loss_cycle_remover.item())
                Cycle_A.append(loss_cycle_adder.item())
                # Idt_R.append(loss_identity_remover.item())
                # Idt_A.append(loss_identity_adder.item())
                # Pred.append(loss_pred.item())
                G.append(loss_G.item())
                D_smoke.append(loss_D_smoke.item())
                D_clean.append(loss_D_clean.item())

            Gan_R = np.average(Gan_R)
            Gan_A = np.average(Gan_A)
            Cycle_R = np.average(Cycle_R)
            Cycle_A = np.average(Cycle_A)
            # Idt_R = np.average(Idt_R)
            # Idt_A = np.average(Idt_A)
            # Pred = np.average(Pred)
            G = np.average(G)
            D_smoke = np.average(D_smoke)
            D_clean = np.average(D_clean)

            real_smokes, real_cleans, pred_smokes, pred_cleans, fake_cleans, rec_smokes, fake_smokes, rec_cleans = \
                map(lambda data: np.concatenate(data, axis=0),
                    [real_smoke_lst, real_clean_lst, pred_smoke_lst, pred_clean_lst, fake_clean_lst, rec_smoke_lst, fake_smoke_lst, rec_clean_lst])

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                        folder_path = self.path + '/results/epoch-{}/sv/'.format(epoch)
                        for np_data in ['real_smokes', 'real_cleans', 'pred_smokes', 'pred_cleans', 'fake_cleans', 'rec_smokes', 'fake_smokes', 'rec_cleans']:
                            np.save(os.path.join(folder_path, np_data+'.npy'), vars()[np_data])

                print_log("Epoch: {0} | G: {1:.4f} D_smoke: {2:.4f} D_clean: {3:.4f}\n".format(
                    epoch + 1, G, D_smoke, D_clean))
                print_log("Gan_R: {0:.4f} Gan_A: {1:.4f} Cycle_R: {2:.4f} Cycle_A: {3:.4f}\n".format(
                    Gan_R, Gan_A, Cycle_R, Cycle_A))
