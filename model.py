import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def l2(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1))

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

# loss functions
def reconstruction_loss(x, x_recon, distribution='gaussian'):
    
    batch_size = x.size(0)  # [256 B, 163]
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'weighted_bernoulli':
         weight = torch.tensor([0.1, 0.9]).to("cuda") # just a label here
         weight_ = torch.ones(x.shape).to("cuda")
         weight_[x <= 0.5] = weight[0]
         weight_[x > 0.5] = weight[1]
         recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none')
         recon_loss = torch.sum(weight_ * recon_loss).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'poisson':
        # print((x - x_recon * torch.log(x)).shape)
        x_recon.clamp(min=1e-7, max=1e7)
        recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
    elif distribution == 'poisson2':
        # layer = nn.Softplus()
        # x_recon = layer(x_recon)
        x_recon = x_recon + 1e-7
        recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
    else:
        raise NotImplementedError

    return recon_loss

def kl_divergence(mu, logvar):
    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    ls = klds.sum(-1).mean()
    return ls

class swapVAE_neural(nn.Module):
    """ swap VAE developed to train neural dataset.
        part of the latent representation is used for clustering, the rest is used to VAE
    """
    def __init__(self, input_size, trial_length, alpha=0.7, beta=0.3, s_dim=64, l_dim=128, hidden_dim = [163, 128], batchnorm=False, learning_rate=5e-4):
        super(swapVAE_neural, self).__init__()

        self.input_size = input_size # number of neurons

        self.s_dim = s_dim
        self.l_dim = l_dim
        self.alpha = alpha
        self.beta = beta
        self.c_dim = int(l_dim - s_dim)
        self.layers_dim = [self.input_size, *hidden_dim] # [163, (163, 128)]

        e_modules = []
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                e_modules.append(nn.BatchNorm1d(trial_length))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], self.l_dim + self.s_dim))
        self.encoder = nn.Sequential(*e_modules)

        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.l_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            if batchnorm:
                d_modules.append(nn.BatchNorm1d(trial_length))
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        # test softplus here
        d_modules.append(nn.Softplus())
        self.decoder = nn.Sequential(*d_modules)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x1, x2):
        # get c and s for x1
        distributions1 = self._encode(x1)
        c1 = distributions1[..., :self.c_dim]
        mu1 = distributions1[..., self.c_dim:self.l_dim]
        logvar1 = distributions1[..., self.l_dim:]
        s1 = reparametrize(mu1, logvar1)
        
        # get c and s for x2
        distributions2 = self._encode(x2)
        c2 = distributions2[..., :self.c_dim]
        mu2 = distributions2[..., self.c_dim:self.l_dim]
        logvar2 = distributions2[..., self.l_dim:]
        s2 = reparametrize(mu2, logvar2)
        
        # create new z1 and z2 by exchanging the content
        z1_new = torch.cat([c2, s1], dim=-1)
        z2_new = torch.cat([c1, s2], dim=-1)
        #### exchange content reconsturction
        x1_recon = (self._decode(z1_new).view(x1.size()))
        x2_recon = (self._decode(z2_new).view(x1.size()))

        #### original reconstruction
        z1_ori = torch.cat([c1, s1], dim=-1)
        z2_ori = torch.cat([c2, s2], dim=-1)
        x1_recon_ori = (self._decode(z1_ori).view(x1.size()))
        x2_recon_ori = (self._decode(z2_ori).view(x1.size()))
        
        return x1, x1_recon, x1_recon_ori, x2, x2_recon, x2_recon_ori, mu1, logvar1, mu2, logvar2, c1, c2

    def loss_function(self, outputs, x=None, distribution_label="poisson"):
        """x1/x2 is augmentation of x, if x is given then we use x to caculate loss"""
        x1, x1_recon, x1_recon_ori, x2, x2_recon, x2_recon_ori, mu1, logvar1, mu2, logvar2, c1, c2 = outputs
        recon1 = reconstruction_loss(x1 if x is None else x, x1_recon, distribution=distribution_label)
        recon2 = reconstruction_loss(x2 if x is None else x, x2_recon, distribution=distribution_label)
        recon1_ori = reconstruction_loss(x1 if x is None else x, x1_recon_ori, distribution=distribution_label)
        recon2_ori = reconstruction_loss(x2 if x is None else x, x2_recon_ori, distribution=distribution_label)
        kl1 = kl_divergence(mu1, logvar1)
        kl2 = kl_divergence(mu2, logvar2)

        l2_loss = l2(c1, c2).mean()
        recon_loss = (recon1 + recon2) / 2
        recon_ori_loss = (recon1_ori + recon2_ori) / 2
        kl_loss = (kl1 + kl2) / 2
        loss = recon_loss + recon_ori_loss + self.alpha* kl_loss + self.beta* l2_loss
        return loss
    
    def train_step(self, x1, x2, x=None):
        self.optimizer.zero_grad()
        vae_outputs = self.forward(x1, x2)
        loss = self.loss_function(vae_outputs, x=x)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def _representation(self, x):
        distributions = self._encode(x)
        mu = distributions[..., :self.l_dim]
        return mu

    def _representation_c(self, x):
        distributions = self._encode(x)
        c = distributions[..., :self.c_dim]
        return c

    def _representation_s(self, x):
        distributions = self._encode(x)
        s = distributions[..., self.c_dim:self.l_dim]
        return s

    def _reconstruct(self, x):
        distributions = self._encode(x)
        mu = distributions[..., :self.l_dim]
        recon = self._decode(mu).view(x.size())
        return recon

"""
distributions1.shape=torch.Size([16, 180, 192])
c1.shape=torch.Size([16, 180, 64])
mu1.shape=torch.Size([16, 180, 64])
logvar1.shape=torch.Size([16, 180, 64])
s1.shape=torch.Size([16, 180, 64])
c2.shape=torch.Size([16, 180, 64])
mu2.shape=torch.Size([16, 180, 64])
logvar2.shape=torch.Size([16, 180, 64])
s2.shape=torch.Size([16, 180, 64])
z1_new.shape=torch.Size([16, 180, 128])
z2_new.shape=torch.Size([16, 180, 128])
x1_recon.shape=torch.Size([16, 180, 182])
x2_recon.shape=torch.Size([16, 180, 182])
"""
