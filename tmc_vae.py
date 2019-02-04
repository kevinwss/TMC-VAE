import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

#from torch.autograd import Variable

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 16
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


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

#-----------------TMC----------------------------

class Node:
    def __init__(self,t, parent = None,left_child = None,right_child = None,name = 0):
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.t = t
        self.z = None
        self.name = name

    def set_t(self,t):
        self.t = t
    def set_z(self,z):
        self.z = z
    
    def set_mean(self,mean):
        self.mean = mean
    def set_var(self,var):
        self.var = var

class TMC:
    def __init__(self, N = 64,a = 1, b = 5, z_dim = 100):

        self.N = N
        self.z_dim = z_dim
        self.leaves = [] #total leaf nodes unchanged
        left = []
        
        for i in range(N):
            self.leaves.append(Node(t=1,name = str(i)))
            
            #left.append(Node(t=1,name = str(i)))
            left = [x for x in self.leaves]
        #bottom up merge to build tree
        
        for j in range(N-1):
            '''
            print("names")
            for n in left:
                print(n.name)
            '''
            total = len(left)
            num1 = np.random.randint(low=0,high = total)
            node1 = left[num1]
            left.remove(node1)

            total = len(left)
            num2 = np.random.randint(low=0,high = total)
            node2 = left[num2]
            left.remove(node2)

            new_node = Node(name= "["+node1.name+"]" +"," + "["+node2.name+"]", t=None,left_child = node1,right_child = node2)
            left.append(new_node)
        
        
       
        self.root = left[0]
        print(self.root.left_child.name,self.root.right_child.name)
        self.root.set_t(0)
        self.root.set_z(np.random.normal(0,1,z_dim))
		
        layer = [self.root]
        new_layer = []
        
        print("built")
        
        #top down layer traversal to assign t and z
        while True:
            #print("iter")
            for node in layer:
                #print("node",node.name,node.left_child.name,node.right_child.name)
				# put non-leaf node to new_layer and check next time
                
                if node.left_child != None:
                    beta_v = np.random.beta(a, b, size=1)[0]
                    #print(beta_v)
                    node.left_child.set_t(node.t + beta_v*(1-node.t))
                    #print("node.z",node.z,"node.left_child.t",node.left_child.t,"node.t",node.t)
                    #print(np.ones(z_dim)*(node.left_child.t - node.t))
                    
                    node.left_child.set_z(np.random.normal(node.z, np.ones(z_dim)*(node.left_child.t - node.t),z_dim)) 
                    node.left_child.set_mean(node.z)
                    node.left_child.set_var(node.left_child.t - node.t)
                 
                    new_layer.append(node.left_child)

                if node.right_child != None:
                    beta_v = np.random.beta(a, b, size=1)

                    node.right_child.set_t(node.t + beta_v*(1-node.t))
                    
                    node.right_child.set_z(np.random.normal(node.z,node.right_child.t - node.t,z_dim))
                    node.right_child.set_mean(node.z)
                    node.right_child.set_var(node.right_child.t - node.t)
					
                    new_layer.append(node.right_child)
                '''
                if node.left_child == None and node.right_child ==None:
                    self.leaves.append(node)
                #   print("leaf")
                '''
            if len(new_layer) == 0:
                break
            
            layer = [x for x in new_layer]
            new_layer = []


    def sample(self,N = 64):
        '''
        self.root.set_z(np.random.normal(0,1,z_dim))

        layer = [self.root]
        new_layer = []
        leaves = []
        #layer level traversal
        while True:
            for node in layer:
                if node.left_child != None:
                    
                    node.left_child.set_z(np.random.normal(node.z,node.left_child.t - node.t))

                    new_layer.append(node.left_child)

                elif node.right_child != None:

                    node.right_child.set_z(np.random.normal(node.z,node.right_child.t - node.t))

                    new_layer.append(node.right_child)
				
				else:
					#leaf node
					pass
					#leaves.append(ndoe)
            if len(new_layer) == 0:
                break
            layer = [x for x in new_layer]
            new_layer = []
'''

        #print(len(self.leaves))
        latent = np.zeros((N,self.z_dim))
        
        means = np.zeros((N,self.z_dim))
        vars = np.zeros((N,self.z_dim))
        
        for i in range(N):
            #latent[i] = self.leaves[i].z  #latent = [0. ,0. , ...] 每个leaf 对应一个数据点        
            latent[i] = np.random.normal(self.leaves[i].mean,self.leaves[i].var,self.z_dim)
            
            means[i] = self.leaves[i].mean
            vars[i] = self.leaves[i].var
        
        return latent,means,vars   #latent = (batch*dim_z)



def KL(q_x,p_x):

    #kl = 0 
    #for i in range(N)：
    # qlog q - qlog p
    p_x = tf.convert_to_tensor(p_x, dtype=tf.float32)
    
    
   
    kl= tf.reduce_sum(q*tf.log(q+0.000000001), 1) - tf.reduce_sum(q*tf.log(p+0.000000001), 1)

    #kl = kl/N

    return kl


def KL_q_tmc(q_means,q_vars,tmc_means,tmc_vars):
    tmc_means = tf.convert_to_tensor(tmc_means, dtype=tf.float32)
    tmc_vars = tf.convert_to_tensor(tmc_vars, dtype=tf.float32)
    
    kl_loss_p_and_tmc = tf.reduce_sum(tf.log(tmc_vars / tf.exp(z_logvar)) + ( (tf.exp(z_logvar))**2+ (z_mu-tmc_means)**2 )/ (2* (tmc_vars)**2) -0.5, 1)
    
    return kl_loss_p_and_tmc
#下面传进去一个 sample了N次的batch p,q


def sample_from_prior():
    pass

#----------------------------------------------------------------

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


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


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)

z_sample = sample_z(z_mu, z_logvar)

#-------tmc-------------------
tmc = TMC(N = mb_size,a=2,b=2,z_dim = z_dim)
tmc_latent, tmc_means, tmc_vars = tmc.sample(N = mb_size)


#-----------------------------

_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_mean( tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1) )
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian

#多个 图片的平均loss
#每个 图片 sample N 次和 q中sample的结果求 KL
#-----------KL loss between q(z|x) and tmc----------------
#kl_loss_p_and_tmc = KL(z_sample,tmc_latent)   #(batch,1)

#kl_loss_p_and_tmc = 0.5 * tf.reduce_sum(tf.log(tmc_vars / tf.exp(z_logvar)) + ( (tf.exp(z_logvar))**2+ (z_mu-tmc_means)**2 )/ (2* (tmc_vars)**2), 1)
kl_loss_p_and_tmc = tf.reduce_mean(  KL_q_tmc(z_mu,tf.exp(z_logvar),tmc_means,tmc_vars) * 0.001 )
#-----------------------------------------------------------

#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)


# VAE loss
vae_loss = recon_loss + kl_loss_p_and_tmc

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, loss ,kl_loss_p_and_tmc_,z_sample_,z_mu_, z_logvar_,recon_loss_ = sess.run([solver, vae_loss, kl_loss_p_and_tmc,z_sample,z_mu, z_logvar,recon_loss], feed_dict={X: X_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print('kl_loss_p_and_tmc:',kl_loss_p_and_tmc_)
        print('recon_loss',recon_loss_)
        #print("z_sample",z_sample_)
        #print("tmc_latent",tmc_latent_)
        
        #print("z_mu",z_mu_)
        #print("z_logvar",z_logvar_)
        print()
        
        #sampled_prior  = np.random.randn(16, z_dim)
        sampled_prior,_,_  = tmc.sample(N=16)   #(16,z_dim)
        print(sampled_prior)
        samples = sess.run(X_samples, feed_dict={z: sampled_prior})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
