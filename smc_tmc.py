import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy
from sklearn.neighbors import KNeighborsClassifier

#from torch.autograd import Variable

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 16
z_dim = 16
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3
#-------------------
L = z_dim -1  #number of stage
C = 20  #particles
a = 1
b = 5


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

def KL_q_tmc(q_means,q_vars,tmc_means,tmc_vars):
    tmc_means = tf.convert_to_tensor(tmc_means, dtype=tf.float32)
    tmc_vars = tf.convert_to_tensor(tmc_vars, dtype=tf.float32)
    
    kl_loss_p_and_tmc = tf.reduce_sum(tf.log(tmc_vars / tf.exp(z_logvar)) + ( (tf.exp(z_logvar))**2+ (z_mu-tmc_means)**2 )/ (2* (tmc_vars)**2) -0.5, 1)
    
    return kl_loss_p_and_tmc
#下面传进去一个 sample了N次的batch p,q

#-------------------------------------------------

def cal_KL_q_tau(q_means,q_vars,tmc_means,tmc_vars):
    #means,vars = tmc.get_mean_var()
    kl_loss_p_and_tmc = tf.reduce_sum(tf.log(tmc_vars**0.5) - z_logvar/2 + ( (tf.exp(z_logvar))+ (z_mu-tmc_means)**2 )/ (2* (tmc_vars)) -0.5, 1)
    return kl_loss_p_and_tmc


class z_node():
    def __init__(self,mean,var):
        self.mean = mean
        self.var = var

class leaf_node():  #used for split
    def __init__(self,mean,z_nodes,var = 1, z = 0,t = 0,left_child = None,right_child = None):
        #self.z_nodes = [z_node(z.mean,z.var) for z in z_nodes] #z index type:[z_node]
        self.z_nodes = z_nodes
        self.mean = mean
        self.var = var 
        self.left_child = left_child #
        self.right_child = right_child #

        self.z = z
        self.t = t

    def update_z_nodes(self):  #update latent vriables
        for i in range(len(self.z_nodes)):
            self.z_nodes[i].mean = self.z
            self.z_nodes[i].var = self.var

    def set_z(self,z):
        self.z = z
    
    def set_mean(self,mean):
        self.mean = mean

    def set_var(self,var):
        self.var = var
       


class TMC():
    def __init__(self,z_dim,root = None):
        self.root = root
        self.z_nodes = [z_node(0,1) for _ in range(z_dim)] #0~z_dim 维护
        self.leaf_nodes = [leaf_node(mean = 0,var = 1, z_nodes = self.z_nodes , z = np.random.normal(0,1),t = 0)] #at first leaf node is only one


    def get_mean_var(self):
        means = np.zeros(z_dim)
        vars = np.zeros(z_dim)
        for i in range(z_dim):
            means[i] = self.z_nodes[i].mean
            vars[i] = self.z_nodes[i].var

        return np.stack([means]*mb_size,axis = 0),np.stack([vars] *mb_size,axis = 0)
    
    def copy(self,tmc2):
        self.z_nodes = [z_node(z_n.mean,z_n.var) for z_n in tmc2.z_nodes]
        self.leaf_nodes = [leaf_node(mean = ln.mean,var = ln.var,z_nodes = self.z_nodes ,z = ln.z,t= ln.t) for ln in tmc2.leaf_nodes]
    
    def sample(self,N = 16):
        latent = np.zeros([N,z_dim])
        for n in range(N):
            for i in range(z_dim):
                latent[n,i] = np.random.normal(self.z_nodes[i].mean,self.z_nodes[i].var)
        
        return latent        
'''
def sample_from_tmc(tmc):
    latent = [0]*z_dim
    i =0
    for node in tmc.nodes:
        latent[i] = sample_gaussian(node.mean,node.var)

    return latent
'''
def num_P(z_dim):#calculate partition numbers
    P = [1,1,1,3] + [0]*(z_dim-4)

    if z_dim%2 ==0:
        e = z_dim
    else:
        e = (z_dim+1)/2

    for n in range(4,z_dim):

        for j in range(1,e):
            P[n] += scipy.misc.comb(z_dim,j)*P[n-j]*P[j] #scipy.misc.comb(N, k, exact=False, repetition=False)
    return P


def generate_next_TMC(tmc_old):
    #generate TMC partition tree strcture according to previous stage
    def normalize_weigths(weights):
        total = np.sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i]/total

        return weights

    def sample_partition(node_to_split,partition_len):# partition_len<=upper(N/2)
        #multinominal without replacement
        total_z_in_leaf = len(node_to_split.z_nodes)
        left = []
        right = []
        '''
        for iter in range(partition_len):
            selected_z = numpy.random.multinomial(z_dim,weights)
            #numpy.random.choice()

            left.append(self.z_nodes(selected_z))
            weights.remove(selected_z)
        '''

        choose = np.random.choice(total_z_in_leaf,partition_len,replace = False) #[2,3,5]
        
        for z_index in range(total_z_in_leaf):
            if z_index in choose:
                left.append(node_to_split.z_nodes[z_index])
            else:
                right.append(node_to_split.z_nodes[z_index])
        #right = (node_to_split.nodes).remove(left)
        return left,right # type : [z_node]

    def sample_node():
        weights = [len(node.z_nodes) for node in tmc.leaf_nodes] #excludes leaf_node with z_dim = 1
        #print("weights",weights)
        
        #weights = [get_P(len(node.z_nodes)) for node in leaf_nodes]
        weigths = normalize_weigths(weights)
        #print("weights",weights)
        #print(np.random.multinomial(1,weights))
        
        node_idx = np.where(np.random.multinomial(1,weights) == 1)[0][0] ####????? sample 1 time, find 1's index
        
        #print("node_idx",node_idx)
        
        sampled_node = tmc.leaf_nodes[node_idx]
        
        #del tmc.leaf_nodes[node_idx] #delete node to be splited
        return sampled_node #type :leaf_node

    #sample a new tmc partition structure
    tmc = TMC(z_dim) # new tmc
    tmc.copy(tmc_old) # copy from old tmc
    
    #print("tmc",tmc)
    #print("tmc leaf_nodes before split",tmc.leaf_nodes)
    
    if len(tmc.leaf_nodes) == 1:

        node = tmc.leaf_nodes[0]
    else:
        node = sample_node()

    partition_len =np.random.choice([x for x in range (1,len(node.z_nodes))],1)  #should according to Combinations
    #print("partition_len",partition_len)
    left,right = sample_partition(node,partition_len[0]) #e.g. left = [1,2] right = [3,4]
    #print("left,right",len(left),len(right))
    
    beta_v = np.random.beta(a, b, size=1)[0]
    node.left_child = leaf_node(mean = node.z, z_nodes = left, t = node.t + beta_v*(1-node.t))
    z_set = np.random.normal(node.z, node.left_child.t - node.t)
    #print("z_set",z_set)
    
    node.left_child.set_z(np.random.normal(node.z, node.left_child.t - node.t))
    node.left_child.set_var(node.left_child.t - node.t)
    node.left_child.update_z_nodes()
    
    
    beta_v = np.random.beta(a, b, size=1)[0]
    node.right_child = leaf_node(mean = node.z, z_nodes = right, t = node.t + beta_v*(1-node.t))
    node.right_child.set_z(np.random.normal(node.z,node.right_child.t - node.t))
    node.right_child.set_var(node.right_child.t - node.t)
    node.right_child.update_z_nodes()
    
    #print("node.right_child.z_nodes mean ", [zn.mean for zn in node.right_child.z_nodes])
    
    tmc.leaf_nodes.remove(node)
    
    '''
    del node
    '''
    
    if len(node.left_child.z_nodes) !=1:
        tmc.leaf_nodes.append(node.left_child)   # add to node to split
    if len(node.right_child.z_nodes) !=1:        # add to node to split
        tmc.leaf_nodes.append(node.right_child)
    
    #print("leaf_nodes after change",tmc.leaf_nodes)
    #print("tmc after",tmc)
    return tmc


def Gaussian(z,mean,var): #z(batch_size,1)
    pi = 3.1415926
    #print(z,mean,var)
    #print(z.shape)
    #mean = np.stack([mean]*mb_size,axis = 0)
    #var = np.stack([var]*mb_size,axis = 0)
    
    #print("mean",mean)
    #print("var",var)
    #ans = ((np.sqrt(2*pi*var))**-1) *np.exp(-0.5*((z-mean)**2/var)) + 0.00000000001
    ans = ((np.sqrt(2*pi*var))**-1) *np.exp(-0.5*((z-mean)**2/var)) + 0.00000000001
    #print("ans",ans)
    return ans

def Cal_P(z,tmc): #(batch_size,z_dim)
    loglikelihood = 0
    i = 0
    #print("tmc.z_nodes mean",[z_n.mean for z_n in tmc.z_nodes])
    #print("tmc.z_nodes var",[z_n.var for z_n in tmc.z_nodes])
    
    likelihood = np.ones(mb_size)
    
    for node in tmc.z_nodes:
        #print(node.mean,node.var)
        #add = (np.log(Gaussian(z[:,i],node.mean,node.var)))
        #print("add",add)
        #likelihood += add
        
        likelihood = likelihood* Gaussian(z[:,i],node.mean,node.var)
        i+=1 
    
    likelihood = np.sum(likelihood)
    #print("likelihood",likelihood)
    return likelihood #(1) sum of a batch of likelihood


def update_TMC(z,P_l,tau_l_star = None, is_first = 1):# z (batch_size,z_dim)
    #tau_l: TMC tree partition structure
    #P = num_P()
    l = 1 #stage
    omega_l = [1 for _ in range(C)] # weights put inside
    omega_lm1 = [1 for _ in range(C)]
    omega_ln = [1 for _ in range(C)]
    j = [1 for _ in range(C)]
    
    while l<L:
        #print("l",l)
        for c in range(0,C): #for different particles
            
            #print("c,l",c,l)
            #print("P_l",len((P_l[c][0]).leaf_nodes))
            
            if c==1:
                #P_l[1] = tau_l_star[c] #1 stage of tau*star

                #P_l[1][l] = tau_l_star[c] ?
                if is_first == 1:
                    P_l[c][l] = generate_next_TMC(P_l[c][l-1])
                else:
                    P_l[c][l] = tau_l_star[l]
                    #print("mean last",[zn.mean for zn in P_l[c][l].z_nodes])
                #P_l[c][l] = tau_l_star[l]

            else:
                P_l[c][l] = generate_next_TMC(P_l[c][l-1])
            
            #print("P_l",P_l)
            #print("P_l[c][l],P_l[c][l-1]",P_l[c][l],P_l[c][l-1])
            ratio = omega_lm1[c]*(Cal_P(z,P_l[c][l])/Cal_P(z,P_l[c][l-1]))
            #ratio = omega_lm1[c]*(Cal_P(z,P_l[c][l-1])/Cal_P(z,P_l[c][l]))
            #print("ratio",ratio)
            omega_l[c] = ratio

            
            omega_lm1[c] = omega_l[c]
        #Normalize weights
        W_l = np.sum(omega_l)
        omega_ln = omega_l/W_l
        
        #if np.sum(omega_ln) != 1:
            #print(np.sum(omega_ln))
            #omega_ln[0] += 1- np.sum(omega_ln)
        #print("omega_l",omega_l)
        #print("W_l",W_l)
        #print("omega_ln",omega_ln,sum(omega_ln))
        #print("omega_ln",np.sum(omega_ln)) 
        j[0] = 1

        for c in range(1,C):  #why?
            
            j[c] = np.where(np.random.multinomial(1,omega_ln) == 1)[0][0]
            #print("j[c]",j[c])
            P_l[c][l] = P_l[j[c]][l] #
            omega_l[c] = W_l/C

        l = l+1 

    
    tau_idx =np.where(np.random.multinomial(1,omega_ln) == 1)[0][0]
    #print("tau_idx",tau_idx)
    l = l -1
    tau_star,tau_l_star = P_l[tau_idx][l],P_l[tau_idx]
    
    #print("mean",[zn.mean for zn in tau_star.z_nodes])
    #print("var",[zn.var for zn in tau_star.z_nodes])
    return tau_star,tau_l_star

def acc(pred,label):
    total = len(pred)
    acc = 0
    
    for i in range(total):
        if pred[i] == label[i]:
            acc+=1
    return acc/total

    
def init_variables():
    # this means stage l = 0

    tau_star = TMC(z_dim)    #一个update过程最后选取的第L个tmc，用于做先.初始化为未分裂的树
    tau_l_star = [TMC(z_dim) for _ in range(L)]  #用于做c=1 的不同stage的值. 初始化为
    
    P_l = [[None for _ in range(L)] for _ in range(C+1)] # particle c 的 L次的tmc 初始化：P_l[c][0] = 未分裂的树 c= 1:C

    for c in range(0,C):
        #print(c)
        P_l[c][0] = TMC(z_dim)
    #P_lm1 = [tmc(z_dim)]*(C+1) #  记录上一次的tmc。 初始化为未分裂的树 暂时先不用这个变量
    #print(P_l)
    #print("P_l",P_l)
    return P_l,tau_star,tau_l_star

#----------------------------------------------------------------

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
tau_mean = tf.placeholder(tf.float32, shape=[None, z_dim]) # mean and var , same in one batch 
tau_var = tf.placeholder(tf.float32, shape=[None, z_dim])


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
    return z_mu, z_logvar   #(batch,z_dim)


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

_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_mean( tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1) )


#多个 图片的平均loss
#每个 图片 sample N 次和 q中sample的结果求 KL

kl_loss_p_and_tmc = tf.reduce_mean(  cal_KL_q_tau(z_mu,tf.exp(z_logvar),tau_mean,tau_var ))
#-----------------------------------------------------------


# VAE loss
vae_loss = recon_loss + kl_loss_p_and_tmc

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

#------train---------
#initrilize tmc
P_l,tau_star,tau_l_star = init_variables()

#get_P = num_P(z_dim) #global variable

#print("get_P:",get_P)

for it in range(1,20000):
    X_mb, _ = mnist.train.next_batch(mb_size)
    #print(tau_star)
    tau_mean_,tau_var_ = tau_star.get_mean_var()
    #print("shape",tau_mean_.shape,tau_var_.shape)
	
    
    _, loss ,kl_loss_p_and_tmc_,z_sample_,z_mu_, z_logvar_,recon_loss_ = sess.run([solver, vae_loss, kl_loss_p_and_tmc,z_sample,z_mu, z_logvar,recon_loss], feed_dict={X: X_mb, tau_mean:tau_mean_, tau_var:tau_var_ })
    #-----------SMC for inferring TMC----------------------
    
    #print("z shape",z_sample_.shape)
    #print('Loss: {:.4}'. format(loss))
    #print('kl_loss_p_and_tmc:',kl_loss_p_and_tmc_)
    #print('recon_loss',recon_loss_)
    if it % 50 == 0:
        #P_l,tau_star,tau_l_star = init_variables()
        for pp in range(1,10):
            P_l,_,_ = init_variables()
            tau_star,tau_l_star = update_TMC(z_sample_,P_l,tau_l_star,is_first = pp)
           
    
    #------------------------------------------------------
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print('kl_loss_p_and_tmc:',kl_loss_p_and_tmc_)
        print('recon_loss',recon_loss_)
        #print("z_sample",z_sample_)
        #print("tmc_latent",tmc_latent_)
        
        #print("z_mu",z_mu_)
        #print("z_logvar",z_logvar_)
        #------------knn---------------
        total_num = 100
        rate = 0.8
        train = total_num*rate
        X_, Y = mnist.train.next_batch(total_num)
        
        z_sample_ = sess.run([z_sample], feed_dict={X: X_})
        z_sample_ = np.array(z_sample_).reshape(total_num,z_dim)
        print("z_sample",z_sample_.shape)
        
        X_train,X_test = z_sample_[:30,:],z_sample_[30:,:]
        Y = [np.where(yy == 1)[0][0] for yy in Y]
        Y_train,Y_test = Y[:30],Y[30:]
        print("Y_test",Y_test)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, Y_train) 
        predict = neigh.predict(X_test)
        print("predict",predict)
        print("acc",acc(predict,Y_test))

        #---------------------------------    
        print()
        
        #sampled_prior  = np.random.randn(16, z_dim)
        sampled_prior= tau_star.sample(N=16)   #(16,z_dim)
        #print(sampled_prior)
        #break
        samples = sess.run(X_samples, feed_dict={z: sampled_prior})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
