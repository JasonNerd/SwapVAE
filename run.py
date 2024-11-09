import torch
from modules.swapvae import swapVAE_neural
spikes = torch.as_tensor(spikes)  # suppose you already have a spike tensor(batch_size, seq_len, num_channel/num_neurons)
num_trials, seq_len, num_channel = spikes.shape
ch_in = 148
sq_in = 140
ratio = 0.5
num_train = int(ratio*num_trials)
spike_train = spikes[num_train:]
# vae = VAE(num_channel, dim_z=num_channel//4, dim_u=seq_len+1, gen_nodes=128)
vae = swapVAE_neural(num_channel, seq_len, batchnorm=True)
epochs = 50
batch_size = 16
num_batch = num_train//batch_size

# train steps: only run once
for e in range(epochs):
    tl = 0.0
    for i in range(num_batch):
        x = spike_train[i*batch_size: (i+1)*batch_size]
        x1 = x.clone()      # channel held in
        x2 = x.clone()      # trial held in
        x1[:,:, ch_in:] = 1e-8
        x2[:, sq_in:,:] = 1e-8
        loss = vae.train_step(x1, x2, x)
        tl+=loss
    print(f"Epoch {e}, loss={loss/num_train}")
torch.save(vae, "swapvae50.pt")

# validation
vae = torch.load("swapvae50.pt")
sp = spike_train.detach().cpu().numpy()
rates = vae._reconstruct(spike_train)
rates = rates.detach().cpu().numpy()
from nlb_tools.evaluation import bits_per_spike
print(bits_per_spike(rates, sp))
# notes
# bits_per_spike is a mertric which can be found at NLB tools


