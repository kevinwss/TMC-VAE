import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3

N = 10 #Num of gaussian


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, N * z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[N * z_dim]))

#----------GMM sigma matrix--------------------
Q_W2_sigma = tf.Variable(xavier_init([h_dim, N * z_dim*z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[N * z_dim*z_dim]))

#----------GMM weights--------------------
Q_W2_a = tf.Variable(xavier_init([h_dim, N]))
Q_b2_a = tf.Variable(tf.zeros(shape=[N]))

def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar_matrix = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    
    a = tf.nn.softmax(tf.matmul(h, Q_W2_a) + Q_b2_a, axis = 1)  #weights for gaussians
    
    z_mu = tf.reshape(z_mu,[mb_size,N,z_dim])   #for each component gaussian, each dimension has a z_mu
    z_logvar_matrix = tf.reshape(z_logvar_matrix, [mb_size, N,z_dim,z_dim])
    #-----GMM-----------
    return a, z_mu , z_logvar_matrix#   a = (batch,N)

    #return z_mu, z_logvar


def sample_z(a, mu, log_var,N):
    #---------GMM--------------
    # a=>np
    # tensor
    #selected = np.random.choice([x for x in range(N)], 1,p=a)
    selected = tf.multinomial(tf.log(a), 1)   #[batch,1]

    #this_mu,this_log_var = mu[selected],log_var[selected]  #select a component
    #this_log_var = shape(z_dim,z_dim)
    print('a',a.shape)
    print('mu',mu.shape)
    print('log_var',log_var.shape)
    print('selected',selected.shape)
    
    z_mu_list = []
    z_logvar_matrix_list = []
    for j in range(mb_size):
        #z_mu_list.append( mu[j,selected[j],:])
        z_mu_list.append( mu[j,selected[j,0],:])
        z_logvar_matrix_list.append(tf.reshape(log_var[j, selected[j,0],:,:],[1,z_dim,z_dim]))
    
    print(z_logvar_matrix_list[0].shape)
    
    this_mu = tf.concat(z_mu_list,0)   #(batch,1, latent)
    this_logvar = tf.concat(z_logvar_matrix_list,0)  #[batch,latent,latent]
    
    
    eps = tf.random_normal(shape=tf.shape(this_mu))
    eps = tf.reshape(eps , [mb_size,z_dim,1])
    print("logvar",this_logvar.shape)
    print("eps",eps.shape)
    this_mu = tf.reshape(this_mu , [mb_size, z_dim,1])
    z = this_mu+tf.matmul( tf.exp(this_logvar) , eps)
    print("z",z.shape)
    
    #this_mu = tf.reshape(this_mu,[mb_size, z_dim])
    return tf.reshape(z,[mb_size, z_dim]),this_mu,this_logvar    #right way to do this?  (batch,latent)
    #return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


def multivariate_gaussian(mu,sigma,N,x): #pdf  #https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    print("x",x.shape)
    print("mu",mu.shape)
    print("sigma",sigma.shape)
    
    mul1 = tf.matmul(x-mu,tf.matrix_inverse(sigma), transpose_a = True )
    exps = tf.exp( -0.5* tf.matmul(mul1, x-mu))
    return ( (((2*3.1415926)**N)*tf.matrix_determinant(sigma))**(-0.5)) *exps



def KL_sample(a,z_mu,z_logvar_matrix,N,sample_iter = 100):
	# a (batch,N) z_mu (batch,N,latent) z_logvar_matrix = (batch,N,latent,latent)
    #KL divergence between two GMM KL(GMM|prior GMM)
    sum = tf.zeros([mb_size,1])
    for i in range(sample_iter):
        '''
        #selected = np.random.choice([x for x in range(N)], 1,p=a)
        selected = tf.multinomial(tf.log(a), 1)   #[batch,1]
        eps = tf.random_normal(shape=tf.shape(this_mu))
        
        z_mu_list = []
        z_logvar_matrix_list = []
        for j in range(batch_size):
            z_mu_list.append( z_mu[j,selected[j],:])
            z_logvar_matrix_list.append(z_logvar_matrix[j, selected[j],:,:])
                
        
        #this_mu, this_logvar = z_mu[selected], z_logvar_matrix[selected]
        this_mu = tf.concat(z_mu_list,0) #(batch,latent)
        this_logvar = tf.concat(z_logvar_matrix_list,0)  #[batch,latent,latent]

        sampled = this_mu + tf.linalg.matmul( tf.exp(this_logvar),eps) # sample from selected gaussian /sample from q 
        #sampeld = (batch,z_dim)
        '''
        x,this_mu,this_logvar = sample_z(a,z_mu,z_logvar_matrix,N)  #(batch,z_dim)
        x = tf.reshape(x,[mb_size,z_dim,1])   
        q = multivariate_gaussian(this_mu,this_logvar,N,x)
        p_mu = tf.zeros([mb_size,z_dim,1])
        p_var = tf.concat([tf.reshape(tf.matrix_diag([1]*z_dim),[1,z_dim,z_dim])]*mb_size, 0)
        
        p_mu = tf.cast(p_mu,tf.float32)
        p_var = tf.cast(p_var,tf.float32)
        
        p = multivariate_gaussian(p_mu,p_var,N,x)
        
        sum += tf.log(q/p)

    kl = sum/sample_iter

    return kl #(batch,1)



# =============================== TRAINING ====================================

a, z_mu, z_logvar = Q(X)

z_sample,_,_ = sample_z(a,z_mu, z_logvar,N)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
kl_loss = KL_sample(a,z_mu,z_logvar,N)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
