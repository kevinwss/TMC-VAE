from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
                    
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
                    
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
                    
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

#--------------parameters------------
N = 5
z_dim = 20
mb_size = 128
#--------------------------------------
def batch_matmul(a,b):
    ans_list = []
    for i in range(mb_size):
        ans_list.append(torch.mm(a[i,:,:],b[i,:,:]).view(1,z_dim,z_dim))
    
    return torch.cat(ans_list,0)

def batch_matmul_21(a,b):
    ans_list = []
    for i in range(mb_size):
        ans_list.append(torch.mm(a[i,:,:],b[i,:].view(z_dim,1)).view(1,z_dim))
    
    return torch.cat(ans_list,0)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, N*z_dim)
        self.fc_Q = nn.Linear(400, N*z_dim*z_dim)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        
        self.fc_a = nn.Linear(400, N)
        self.fc_e = nn.Linear(400, N*z_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        
        mu = self.fc_mu(h1)
        logQ = self.fc_Q(h1)
        
        softmax = torch.nn.Softmax()
        a = softmax(self.fc_a(h1))
        
        eigen = self.fc_e(h1)
        
        mu = mu.view(-1,N,z_dim)
        Q = torch.exp(logQ.view(-1,N,z_dim,z_dim))
        
        
        eigen = torch.exp(eigen.view(-1,N,z_dim))
        
        return mu,Q, a ,eigen # 
        #return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return eps.mul(std).add_(mu)

#--------------- GMM -----------------------
    def sample(self, mu, Q,a,eigen):
        #print(a)
        idx= 0
        
        m = torch.distributions.Categorical(a)
        idx = m.sample()
        
        this_mu = torch.cat([mu[i,idx[i],:].view(1,z_dim) for i in range(mb_size)],0)# (batch,z_dim)
        #this_eigen = eigen[:,idx,:].view(-1,z_dim)
        this_eigen = torch.cat([eigen[i,idx[i],:].view(1,z_dim) for i in range(mb_size)],0)
        #this_Q = Q[:,idx,:,:].view(-1,z_dim,z_dim)  #(batch,1,latent,latent)
        this_Q  = torch.cat([Q[i,idx[i],:,:].view(1,z_dim,z_dim) for i in range(mb_size)],0)
        eps =  torch.randn(mb_size,z_dim)
        
        diag_list = []
        for i in range(mb_size):
            diag_list.append(torch.diag(this_eigen[i,:]).view(1,z_dim,z_dim))
        
        m_diag = torch.cat(diag_list,0)       #(batch,20)
        mul1 = batch_matmul(this_Q,m_diag)
        var = batch_matmul(mul1, torch.inverse(this_Q))

        z = batch_matmul_21(var,eps) + this_mu  #(batch,z_dim)
        
        return z
#-------------------------------------------

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, Q, a, eigen= self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logvar)
        logvar = Q
        z = self.sample(mu, Q,a,eigen)
        
        return self.decode(z), mu, logvar,a,eigen


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#-------------------------------------
def KL_cal(a,mu,Q,eigen):
    #print("a",a)

    KL = 0
    for idx in range(N):
        #idx= 0
        this_mu = mu[:,idx,:].view(-1,z_dim)
        this_eigen = eigen[:,idx,:].view(-1,z_dim)
        this_Q = Q[:,idx,:,:].view(-1,z_dim,z_dim)  #(batch,1,latent,latent)
        diag_list = []
        
        for i in range(mb_size):
            diag_list.append(torch.diag(this_eigen[i,:]).view(1,z_dim,z_dim))
                
        m_diag = torch.cat(diag_list,0)       #(batch,20)
        mul1 = batch_matmul(this_Q,m_diag)
        var = batch_matmul(mul1, torch.inverse(this_Q))

        p_mu = torch.zeros([mb_size,z_dim])
        p_sigma = torch.cat([torch.diag(torch.ones([z_dim],dtype=torch.float32)).view([1,z_dim,z_dim])]*mb_size, 0)
        
        
        ans = []
        for i in range(mb_size):
            _sigma0,_sigma1,_mu0,_mu1 = var[i,:,:],p_sigma[i,:,:],this_mu[i,:].view(1,z_dim),p_mu[i,:].view(1,z_dim)
            
            #print(torch.inverse(_sigma1))
            kl1 = torch.trace(torch.inverse(_sigma1)*_sigma0)
            
            #print(_mu0.shape)
            #print((_mu1-_mu0).transpose(1,0).shape)
            #print(_sigma1.shape)
            mul1 = torch.mm((_mu1-_mu0),torch.inverse(_sigma1)) #(1,20)
            d1 = torch.det(_sigma1)
            d0 = torch.det(_sigma0)
            d = torch.log(d1/d0)
            kl2 = torch.mm(mul1, ( _mu1-_mu0).transpose(1,0)) - N + d
            
            kl = 0.5*(kl1+kl2)
            with torch.no_grad():
                ans.append(kl*a[i,idx])
    
        KL += torch.sum(torch.cat(ans,0))
    return KL

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, Q,a,eigen):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
   # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    KLD = KL_cal(a,mu,Q,eigen)
    #print(KLD)
    #print(BCE)
    #return BCE + KLD
    return BCE + KLD,KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        #print("data",data.shape)
        #print("data",data.shape[0])
        
        #global
        mb_size = int(data.shape[0])
        
        optimizer.zero_grad()
        #recon_batch, mu, logvar = model(data)
        recon_batch, mu, Q,a,eigen = model(data)
        
        loss,KLD = loss_function(recon_batch, data, mu, Q,a,eigen)
        #print("loss",loss)
        #print("KLD",KLD)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            print("KLD",KLD.item()/len(data))
            #print("a",a)
               
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(batch_idx) + '.png')
                      
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
