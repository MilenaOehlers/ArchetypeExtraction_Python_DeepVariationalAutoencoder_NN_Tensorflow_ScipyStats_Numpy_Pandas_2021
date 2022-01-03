#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:25:27 2020

Code adapted from Keller, Sebastian Mathias, et al. "Learning extremal representations with deep archetypal analysis." International Journal of Computer Vision 129.4 (2021): 805-820
for materials science data (original code was built for image analysis)
@author: Keller, Sebastian Mathias, et al., oehlers
"""
import numpy as np
from scipy.linalg import solve
import scipy as sp
import tensorflow as tf
tfd = tf.contrib.distributions 
import os
import pandas as pd
import tensorflow as tf
from nets_benchmarking.func_collection import get_zfixed

    
##########################################################################################################
class lib_at:
    def __init__(self):
        pass
    
    def centralize_matrix(M):
        """
        See https://en.wikipedia.org/wiki/Centering_matrix
        :param M:
        :return:
        """
        n, p = M.shape
        Q = np.full((n, n), -1 / n)
        Q = Q + np.eye(n)
        M = np.dot(np.dot(Q, M), Q)
        return M
        
    def greedy_min_distance(z_f, z_true_mu):
        """
        Calculates the mean L2 Distance between found Archetypes and the true Archetypes (in latent space).
        1. Select the 2 vector with smallest pairwise distance
        2. Calculate the euclidean distance
        3. Remove the 2 vectors and jump to 1.
        :param z_f:
        :param z_true_mu:
        :return: mean loss
        """
        loss = []
        dist = sp.spatial.distance.cdist(z_f, z_true_mu)
        for i in range(z_f.shape[0]):
            z_fixed_idx, z_true_idx = np.unravel_index(dist.argmin(), dist.shape)
            loss.append(dist[z_fixed_idx, z_true_idx])
            dist = np.delete(np.delete(dist, z_fixed_idx, 0), z_true_idx, 1)
        return loss
        
    def create_z_fix(dim_latent_space):
        """
        Creates Coordinates of the Simplex spanned by the Archetypes.
    
        The simplex will have its centroid at 0.
        The sum of the vertices will be zero.
        The distance of each vertex from the origin will be 1.
        The length of each edge will be constant.
        The dot product of the vectors defining any two vertices will be - 1 / M.
        This also means the angle subtended by the vectors from the origin
        to any two distinct vertices will be arccos ( - 1 / M ).
    
        :param dim_latent_space:
        :return:
        """
        return get_zfixed(dim_latent_space)

        
    def barycentric_coords(n_per_axis=5):
        """
        Creates coordinates for the traversal of 3 Archetypes (i.e. creates the a weights)
        :param n_per_axis:
        :return: [weights, n_perAxis]; weights has shape (?, 3)
        """
        weights = np.zeros([int((n_per_axis * (n_per_axis + 1)) / 2), 3])
    
        offset = np.sqrt(3 / 4) / (n_per_axis - 1)
        A = np.array([[1.5, 0, 0], [np.sqrt(3) / 2, np.sqrt(3), 0], [1, 1, 1, ]])
        cnt = 0
        innerCnt = 0
        for i in np.linspace(0, 1.5, n_per_axis):
            startX = i
            startY = cnt * offset
    
            if n_per_axis - cnt != 1:
                stpY = (np.sqrt(3) - 2 * startY) / (n_per_axis - cnt - 1)
            else:
                stpY = 1
            for j in range(1, n_per_axis - cnt + 1):
                P_x = startX
                P_y = startY + (j - 1) * stpY
                b = np.array([P_x, P_y, 1])
                sol = solve(A, b)
    
                out = np.abs(np.around(sol, 6))
                weights[innerCnt, :] = out
                innerCnt += 1
            cnt += 1
    
        return [weights, n_per_axis]
    
##########################################################################################################
## lib_vae code:
def share_variables(func):
    """
    Wrapper for tf.make_template as decorator.
    :param func:
    :return:
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)

class lib_vae:
    def __init__(self):
        """
        Contains the Network Architectures.
        """

    
    
    def build_prior(dim_latentspace):
        """
        Creates N(0,1) Multivariate Normal prior.
        :param dim_latentspace:
        :return: mvn_diag
        """
        mu = tf.zeros(dim_latentspace)
        rho = tf.ones(dim_latentspace)
        mvn_diag = tfd.MultivariateNormalDiag(mu, rho)
        return mvn_diag
    
    def dirichlet_prior(dim_latentspace, alpha=1.0):
        """
    
        :param dim_latentspace:
        :param alpha:
        :return:
        """
        nATs = dim_latentspace + 1
        alpha = [alpha] * nATs
        dist = tfd.Dirichlet(alpha)
        return dist
    
    def build_encoder_basic(dim_latentspace, z_fixed, version='original'):
        """
        Basic DAA Encoder Architecture. Not actually used but more for showcasing the implementation.
        :param dim_latentspace:
        :param z_fixed:
        :return:z_predicted, mu_t, sigma_t, t
        """
        
        @share_variables
        def encoder(data):
            if version=='luigi':
                print("luigis version to be added")
                
            if version in ['original','milena']:
                nAT = dim_latentspace + 1
        
                #x = tf.layers.flatten(data)
                net = tf.layers.dense(data, 200, tf.nn.relu)
                net = tf.layers.dense(net, 100)
                mean_branch, var_branch = net[:, :50], net[:, 50:]
        
                # Weight Matrices
                weights_A = tf.layers.dense(mean_branch, nAT, tf.nn.softmax)
                weights_B_t = tf.layers.dense(mean_branch, nAT)
                weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)
        
                # latent space parametrization
                mu_t = tf.matmul(weights_A, z_fixed)
                sigma_t = tf.layers.dense(var_branch, dim_latentspace, tf.nn.softplus)
                t = tfd.MultivariateNormalDiag(mu_t, sigma_t)
                
                # predicted archetypes
                z_predicted = tf.matmul(weights_B, mu_t)
        
            return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}
        
        return encoder
    
    def build_decoder(n_feats, n_targets, trainable_var=True, version='original'):
        """
         Builds Decoder for jaffe data
        :param n_feats:
        :param num_labels:
        :param trainable_var: Make the variance of the decoder trainable.
        :return:
        """
        
        @share_variables
        def decoder(latent_code):
            ## comment by MO: 
            # in original JAFFE version, var of feat. space (of pixel darkness) was 
            # set to be trainable but equal for all pixels. 
            # in our case, the features are fundamentally different to each other 
            # and if a variance is learned by the network, it should be one for each
            # feature -> thats why above, in var = ... [.]*(n_targets+n_feats) was added
            activation = tf.nn.relu
            
            if version=='luigi':
                print("luigis version to be added")
            if version=='milena':
                def decod_func(units):
                    var = tf.Variable(initial_value=[1.0]*units, trainable=trainable_var)
        
                    x = tf.layers.dense(latent_code, units=200, activation=activation)
                    for i in np.arange(3):
                        x = tf.layers.dense(x, units=units, activation=activation)
                    x_hat = tf.layers.dense(x, units=units, activation=tf.nn.sigmoid)
                    shape = x_hat.shape
                    x_m = tf.convert_to_tensor(x_hat) #tf.math.scalar_mul(1.0,x_hat)
                    x_mean = x_m #tf.matmul(x_hat,x_hat)
                    x_hat = tfd.Normal(x_hat, var)
                    x_hat = tfd.Independent(x_hat, 2)
                    #print('x_hat params:',x_hat.mean, x_hat.variance)
                    return x_hat, x_mean, x_m
                
                x_hat, x_mean, x_m = decod_func(units=n_feats)
                side_info, y_mean, y_m = decod_func(units=n_targets)
            
            if version=='original':
                var = tf.Variable(initial_value=1.0, trainable=trainable_var)
    
        
                activation = tf.nn.relu
                units = 49
                x = tf.layers.dense(latent_code, units=units, activation=activation)
                x = tf.layers.dense(x, units=units, activation=activation)
                
                #x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, units=np.prod(n_feats), activation=activation)
                x = tf.layers.dense(x, units=np.prod(n_feats), activation=tf.nn.sigmoid)
    
                x_mean = tf.convert_to_tensor(x)
                x_hat = tfd.Normal(x, var)
                x_hat = tfd.Independent(x_hat, 2)
        
                side_info = tf.layers.dense(latent_code, 200, tf.nn.relu)
                side_info = tf.layers.dense(side_info, n_targets, tf.nn.sigmoid) * 5
                y_mean = tf.convert_to_tensor(side_info)
                
            return {"x_hat": x_hat, "side_info": side_info, "x_mean": x_mean, "y_mean": y_mean}
    
        return decoder
    
##########################################################################################################
## daa_JAFFE code:

def execute(data, version = 'original', epochs=100,at_loss_factor=8.0, target_loss_factor=8.0,recon_loss_factor=4.0,kl_loss_factor=4.0, anneal=0):
    # If error message "Could not connect to any X display." is issued, uncomment the following line:
    #os.environ['QT_QPA_PLATFORM']='offscreen'
    x_train_feat = data['train_feat']
    x_train_targets = data['train_targets']
    
    # NN settings
    gpu = '0'
    learning_rate = 1e-4
    n_epochs = epochs           # 5001 in original code
    batch_size = 50
    dim_latentspace =2
    seed = None
    n_targets = x_train_targets.shape[1]
    trainable_var = False

    # Different settings for the prior
    vamp_prior = False
    dir_prior = False
    vae = False

    assert not (dir_prior and vamp_prior), "The different priors are mutually exclusive."
    assert 0 < n_targets <= 5, "Choose up to 5 targets."
    
    # GPU targets
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)
    
    def build_loss(version='original'):
        """
        Build all the required losses for the Deep Archetype Model.
        :return: archetype_loss, target_loss, likelihood, divergence, elbo
        """
        likelihood = tf.reduce_sum(x_hat.log_prob(data)) 
        if vamp_prior or dir_prior:
            q_sample = t_posterior.sample(50)
            divergence = tf.reduce_mean(encoded_z_data["p"].log_prob(q_sample) - prior.log_prob(q_sample))
        else:
            divergence = tf.reduce_mean(tfd.kl_divergence(t_posterior, prior))

        if not vae:
            archetype_loss = tf.losses.mean_squared_error(z_predicted, z_fixed)
        else:
            archetype_loss = tf.constant(0, dtype=tf.float32)
            
        # Sideinformation Reconstruction loss
        if version=='milena':
            target_loss = tf.reduce_sum(y_hat.log_prob(side_information)) 
        if version in ['original','luigi']:
            target_loss = tf.losses.mean_squared_error(side_information, y_hat)         
            
        elbo = tf.reduce_mean(
            recon_loss_factor * likelihood
            + target_loss_factor * target_loss 
            - at_loss_factor * archetype_loss
            - kl_loss_factor * divergence
        )

        return archetype_loss, target_loss, likelihood, divergence, elbo
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Japanese Face Expressions

    n_total_samples, n_feats = x_train_feat.shape[0], x_train_feat.shape[1]
    n_batches = int(n_total_samples / batch_size)

    def get_batch(batch_size, x_train_feat, x_train_targets, shuffle=True):
        x_train = pd.DataFrame(np.concatenate([x_train_feat,x_train_targets.reshape([n_total_samples,n_targets])],axis=1))
        x_train_sampled = x_train.sample(n=batch_size) if shuffle else x_train
        x_train_feat_sampled, x_train_targets_sampled = x_train_sampled.iloc[:,:-n_targets],x_train_sampled.iloc[:,-n_targets:] #.reshape((x_train_sampled[0],n_targets))
        return x_train_feat_sampled, x_train_targets_sampled
    
    def get_next_batch(batch_size):
        """
        Helper function for getting mini batches.
        :param batch_size:
        :return: mb_x, mb_y
        """
        mb_x, mb_y = get_batch(batch_size, x_train_feat, x_train_targets)
        return mb_x, mb_y

    all_feats, all_targets = get_batch(n_total_samples,x_train_feat, x_train_targets)
    
    some_feats = all_feats[:batch_size]
    some_targets = all_targets[:batch_size]
    all_imgs_ext = np.vstack((all_feats, all_feats[:batch_size]))

    ####################################################################################################################
    # ########################################### Data Placeholders ####################################################
    data = tf.placeholder(tf.float32, [None, n_feats], 'data')
    side_information = tf.placeholder(tf.float32, [None, n_targets], 'targets')
        
    ####################################################################################################################
    # ########################################### Model Setup ##########################################################
    z_fixed_ = lib_at.create_z_fix(dim_latentspace)
    z_fixed = tf.cast(z_fixed_, tf.float32)
    
    encoder_net = lib_vae.build_encoder_basic(dim_latentspace=dim_latentspace, z_fixed=z_fixed, version=version)
    decoder = lib_vae.build_decoder(n_feats=n_feats, n_targets=n_targets, trainable_var=trainable_var, version=version)

    prior = lib_vae.build_prior(dim_latentspace)
    
    encoded_z_data = encoder_net(data)
    try:
        z_predicted, mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["z_predicted", "mu", "sigma", "p"]]
    except KeyError:
        assert vae
        mu_t, sigma_t, t_posterior = [encoded_z_data[key] for key in ["mu", "sigma", "p"]]
    decoded_post_sample = decoder(t_posterior.sample())
    x_hat, y_hat, x_mean, y_mean = decoded_post_sample["x_hat"], decoded_post_sample["side_info"],decoded_post_sample["x_mean"], decoded_post_sample["y_mean"]
    
    # Build the loss
    archetype_loss, target_loss, likelihood, kl_divergence, elbo = build_loss(version)

    # Build the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)
 
    # Specify what is to be logged.
    tf.summary.scalar(name='elbo', tensor=elbo)
    tf.summary.scalar(name='archetype_loss', tensor=archetype_loss)
    tf.summary.scalar(name='target_loss', tensor=target_loss)
    tf.summary.scalar(name='likelihood', tensor=likelihood)
    tf.summary.scalar(name='kl_divergence', tensor=kl_divergence)

    summary_op = tf.summary.merge_all()

    def create_latent_df():
        """
        Create pandas DF with the latent mean coordinates + targets of the data.
        :return: Dataframe  pd.DataFrame(array_all, columns=['targets','ldim0',..., 'ldimN'])
        
        def get_var(var,feed_what_with_what_dict):
            var_stacked = None
            for i in range(all_imgs_ext.shape[0] // batch_size):
                min_idx = i * batch_size
                max_idx = (i + 1) * batch_size
                tmp_mu= sess.run(var, feed_dict={data: all_feats[min_idx:max_idx],
                                                   side_information: all_targets[min_idx:max_idx]})
                var_stacked = np.vstack((test_pos_mean, var)) if test_pos_mean is not None else tmp_mu
            
            var_stacked = var_stacked[:all_feats.shape[0]]
            array_all = np.hstack((test_pos_mean, all_targets))
            cols_dims = [f'ldim{i}' for i in range(dim_latentspace)]
            df = pd.DataFrame(array_all, columns=cols_dims + ['target'])
            return df
        """
        test_pos_mean = None
        test_pos_sigma = None
        for i in range(all_imgs_ext.shape[0] // batch_size):
            min_idx = i * batch_size
            max_idx = (i + 1) * batch_size
            tmp_mu= sess.run(mu_t, feed_dict={data: all_feats[min_idx:max_idx],
                                               side_information: all_targets[min_idx:max_idx]})
            test_pos_mean = np.vstack((test_pos_mean, tmp_mu)) if test_pos_mean is not None else tmp_mu
            
            tmp_sigma= sess.run(sigma_t, feed_dict={data: all_feats[min_idx:max_idx],
                                               side_information: all_targets[min_idx:max_idx]})
            test_pos_sigma = np.vstack((test_pos_sigma, tmp_sigma)) if test_pos_sigma is not None else tmp_sigma
            
        test_pos_mean = test_pos_mean[:all_feats.shape[0]]
        test_pos_sigma = test_pos_sigma[:all_feats.shape[0]]
        array_all = np.hstack((test_pos_mean,test_pos_sigma, all_targets))
        
        cols_dims = [f'mu{i}' for i in range(dim_latentspace)]
        cols_sigma = [f'sigma{i}' for i in range(dim_latentspace)]
        cols_targets = [f'target{i}' for i in range(n_targets)]
        
        df = pd.DataFrame(array_all, columns=cols_dims + cols_sigma + cols_targets)
        
        return df
    
    def extract_xy_hat(df):
        tmp_xmean = sess.run(x_mean, feed_dict={data: all_feats,
                                               side_information: all_targets})
        tmp_ymean = sess.run(y_mean, feed_dict={data: all_feats,
                                               side_information: all_targets})
        return tmp_xmean, tmp_ymean

    ####################################################################################################################
    ############################################# Training Loop ########################################################
    #saver = tf.train.Saver()
    step = 0
    sess.run(tf.global_variables_initializer())

    #writer = tf.summary.FileWriter(logdir=TENSORBOARD_DIR, graph=sess.graph)
    kl_loss_max = kl_loss_factor
    for epoch in range(n_epochs):
        if (epoch+1)/n_epochs in np.linspace(0,1,11): print('epoch no {} of {}'.format(epoch+1,n_epochs))
        for b in range(n_batches):
            mb_x, mb_y = get_next_batch(batch_size)
            
            
            if anneal ==1 and epoch <= 100:
                kl_loss_factor = 0
                kl_loss_factor += epoch/100 * kl_loss_max
            
            feed_train = {data: mb_x, side_information: mb_y}
            sess.run(optimizer, feed_dict=feed_train)
            step += 1

        
        # evaluate metrics on some feats; NOTE that this is no real test set
        tensors_test = [summary_op, elbo,
                        likelihood,
                        kl_divergence, archetype_loss, target_loss]
        feed_test = {data: some_feats, side_information: some_targets}
        summary, test_total_loss, test_likelihood, test_kl, test_atl, test_targetl = sess.run(tensors_test,
                                                                                                    feed_test)
        
    print("Model Trained!")
    df = create_latent_df()
    
    df_targets = df[[col for col in df.columns if 'target' in col]]
    df_features = df[[col for col in df.columns if 'mu' in col]]
    df_sigma = df[[col for col in df.columns if 'sigma' in col]]
    
    xhat,yhat = extract_xy_hat(df.iloc[:,:-n_targets])
    if n_targets == 1: yhat = yhat.reshape(n_total_samples)
    
    def asarrays(lst):
        return [np.array(i) for i in lst]
    result_key = (version, at_loss_factor,target_loss_factor,recon_loss_factor,kl_loss_max,anneal)
    result_df = pd.DataFrame({'features': [np.array(x_train_feat), tuple([np.array(df_features),np.array(df_sigma)]), np.array(xhat)],
                           'target_color': asarrays([x_train_targets, df_targets, yhat])},
                        index=['real space', 'latent space', 'reconstructed real space'])
    result_dict = {result_key : result_df }

    return result_dict
