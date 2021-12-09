import argparse
import itertools
import torch
import torch.nn as nn
import pickle

from torch.autograd import Variable
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--shift', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='total epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='rescale size')
parser.add_argument('--input_c', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_c', type=int, default=3, help='number of channels of output data')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle dataset or not')
parser.add_argument('--save_models', type=bool, default=False, help='save models or not')
parser.add_argument('--save_losses', type=bool, default=False, help='save losses or not')
parser.add_argument('--lmbda', type=float, default=10.0, help='lambda coefficient for adversarial loss')
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
RESCALE_SIZE = args.size 

input_c = args.input_c
output_c = args.output_c

losses_gan = []
losses_a = []
losses_b = []


def train(data, G_AB, G_BA, D_A, D_B, n_epochs=200, batch_size=1, lr=0.0002, lmbda=10.0, save_models=True, 
          save_losses=True, shuffle=True, shift=0, input_c=3, output_c=3):
    global losses_adv, losses_a, losses_b, DEVICE, RESCALE_SIZE
    losses_adv = []
    losses_a = []
    losses_b = []

    target_real = Variable(torch.Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(torch.Tensor(batch_size).fill_(0.0), requires_grad=False)

    

    G_AB.to(DEVICE)
    G_BA.to(DEVICE)
    D_A.to(DEVICE)
    D_B.to(DEVICE)
    target_real = target_real.to(DEVICE)
    target_fake = target_fake.to(DEVICE)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)


    generative_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    id_loss = nn.L1Loss()

    
    optimizer_G = torch.optim.Adam(params=itertools.chain(G_AB.parameters(), G_BA.parameters()), 
                                    lr=lr, betas=(0.5, 0.999))
    optimizer_A = torch.optim.Adam(params=D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_B = torch.optim.Adam(params=D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, 
                                                       lr_lambda=LambdaLR(n_epochs-shift, 0, n_epochs/2-shift).step)
    lr_scheduler_A = torch.optim.lr_scheduler.LambdaLR(optimizer_A, 
                                                         lr_lambda=LambdaLR(n_epochs, 0, n_epochs/2-shift).step)
    lr_scheduler_B = torch.optim.lr_scheduler.LambdaLR(optimizer_B, 
                                                         lr_lambda=LambdaLR(n_epochs, 0, n_epochs/2-shift).step)
    

    fake_A_buffer = Buffer()
    fake_B_buffer = Buffer()

    with tqdm(desc="epoch", total=n_epochs-shift) as pbar_outer:
        for epoch in range(shift+1, n_epochs+shift+1):
            lg = 0
            la = 0
            lb = 0
            k = 1
            for batch in dataloader:
                real_A = batch['A'].to(DEVICE)
                real_B = batch['B'].to(DEVICE)
                #skipping images with wrong number of channels
                if real_A.shape[1] != input_c or real_B.shape[1] != input_c:
                    continue

                optimizer_G.zero_grad()
                optimizer_A.zero_grad()
                optimizer_B.zero_grad()
                k += 1

                # identity loss
                id_B = G_AB(real_B)
                loss1 = id_loss(id_B, real_B) * (lmbda / 2)
                id_A = G_BA(real_A)
                loss2 = id_loss(id_A, real_A) * (lmbda / 2)

                fake_B = G_AB(real_A)
                detect_fake_B = D_B(fake_B)
                loss3 = generative_loss(detect_fake_B, target_real)
                fake_A = G_BA(real_B)
                detect_fake_A = D_A(fake_A)
                loss4 = generative_loss(detect_fake_A, target_real)

                # cycle loss
                cycle_A = G_BA(fake_B)
                loss5 = cycle_loss(cycle_A, real_A) * lmbda
                cycle_B = G_AB(fake_A)
                loss6 = cycle_loss(cycle_B, real_B) * lmbda

                loss_adv = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                loss_adv.backward()
                lg += loss_adv.item()

                optimizer_G.step()

                # adversarial loss A
                pred_real = D_A(real_A)
                loss7 = generative_loss(pred_real, target_real)

                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = D_A(fake_A.detach())
                loss8 = generative_loss(pred_fake, target_fake)

                loss_a = (loss7 + loss8) * 0.5
                loss_a.backward()
                la += loss_a.item()

                optimizer_A.step()

                # adversarial loss B
                pred_real = D_B(real_B)
                loss9 = generative_loss(pred_real, target_real)

                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = D_B(fake_B.detach())
                loss10 = generative_loss(pred_fake, target_fake)

                loss_b = (loss9 + loss10) * 0.5
                loss_b.backward()
                lb += loss_b.item()
                
                optimizer_B.step()

            pbar_outer.update(1)

            lr_scheduler_G.step()
            lr_scheduler_A.step()
            lr_scheduler_B.step()

            losses_adv.append(lg / k)
            losses_a.append(la / k)
            losses_b.append(lb / k)
            wandb.log({'loss_adv_a_b': la / k,
                       'loss_adv_b_a': lb / k,
                       'loss_cycle_consistancy': lg / k})
            print(epoch, losses_adv[-1], losses_a[-1], losses_b[-1], '---------', sep='\n')
            torch.save(G_AB.state_dict(), f'{args['dataroot']}/saved_models/G_AB_{dataset}')
            torch.save(G_BA.state_dict(), f'{args['dataroot']}/saved_models/G_BA_{dataset}')
            torch.save(D_A.state_dict(), f'{args['dataroot']}/saved_models/D_A_{dataset}')
            torch.save(D_B.state_dict(), f'{args['dataroot']}/saved_models/D_B_{dataset}')

            if real_A.shape[1] == input_c:
                with torch.no_grad():
                    plot(real_A, G_AB(real_A), G_BA(G_AB(real_A)))
