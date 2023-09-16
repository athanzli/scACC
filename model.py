import sys
import gc
import torch
import sklearn.cluster
import sklearn.ensemble
import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from IPython.display import clear_output # NOTE

import umap
from torch import nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

from .utils import *

class ClusterLayer(torch.nn.Module):
    def __init__(self, n_clusters, latent_dim):
        super().__init__()
        self.cluster_centers = torch.nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        self.n_clusters = n_clusters

    def forward(self, z):
        r"""
        Update cluster centers and calculate q.

        NOTES:
            Recursive computations: You may be computing the gradients recursively in a loop, either implicitly or explicitly, which may be causing the computational graph to be traversed more than once.
        """
        q = calc_q(z, self.cluster_centers)
        p = target_distribution(q.data)
        return q, p

class scACC():
    r"""
    The scACC model.

    Args:
        latent_dim: latent dimension
        n_clusters: number of clusters
        lam2: for kl-div loss
        lam3: for entropy loss
        dropout: for AE
        device: ...
        alpha: ...
    """
    def __init__(self,
                 layers=None,
                 n_clusters=30, 
                 lam1=1,
                 lam2=1, 
                 lam3=1, 
                 device='cpu',
                 alpha=1,
                 drop_rate=.2, 
        ):

        self.layers = layers
        self.n_clusters = n_clusters
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.drop_rate = drop_rate
        self.alpha = alpha
        self.device = torch.device(device)
        set_device(device)

    # def __call__(self, x):
    #     r"""
    #     Returns:
    #         x_bar: reconstructed 
    #         q: clustering Q matrix 
    #         z: embedding
    #     """
    #     x = torch.tensor(x).float().to(self.device)
    #     z, x_bar = self.ae(x)
    #     q = calc_q(z, self.clusters, self.alpha)
    #     return x_bar, q, z

    class _AE(nn.Module):
        r"""
        Autoencoder module of scACC.
        """

        def __init__(self, layers=[1024,512,256,128], drop_rate=.2, device='cpu', n_clusters=30):
            super().__init__()
            self.drop_rate = drop_rate
            self.device = device
            
            self.layers = layers
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(drop_rate)

            # Encoder
            encoder_layers = []
            for i in range(len(layers) - 1):
                encoder_layers.append(nn.Linear(layers[i], layers[i+1]))
                encoder_layers.append(self.act)
                if i != len(layers) - 2: # no dropout to the latent layer
                    encoder_layers.append(self.dropout)
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            for i in range(len(layers) - 1, 0, -1):
                decoder_layers.append(nn.Linear(layers[i], layers[i-1]))
                decoder_layers.append(self.act)
            self.decoder = nn.Sequential(*decoder_layers)

            self.cluster_layer = ClusterLayer(n_clusters, layers[-1])

        def forward(self, x):
            x = torch.tensor(x).float().to(self.device)
            z = self.encoder(x)
            q, _ = self.cluster_layer(z) # NOTE, is use q = calc_q(z, self.cluster_layer.cluster_centers.data) then cluster_centers will not update b/c it is not in the comp graph.
            x_bar = self.decoder(z)
            return x_bar, q

    def init_ae(self):
        r"""
        A helper function for creating the ae attribute in order for pretrained
        data loading.
        """
        self.ae = self._AE(
            layers=self.layers,
            drop_rate=self.drop_rate,
            device=self.device,
            n_clusters=self.n_clusters).to(self.device)
    
    def train(self,
            X_train, 
            y_train, 
            epoch_pretrain=15,
            epoch_train=15, # do not train too many epochs, it may improve accuracy but will not make sense biologically
            batch_size=512,
            lr_pretrain=1e-3,
            lr_train=1e-3,
            lam1=None,
            lam2=None,
            lam3=None,

            early_stopping=True,
            lr_decay=False,

            require_pretrain_phase=True,
            require_train_phase=True, 
            evaluation=False,
            plot_evaluation=False, # plot metics per epoch
            id_train=None, X_val=None, y_val=None, id_test=None, # for printing out metrics per epoch
            fold_num=None,

            c_true_labels_dummy=None,
        ):
        r"""
        Train scACC model, including pretraining phase and training phases

        Args:
            X_train: cell-by-genes data matrix. rows are cells, columns are genes,
                and entries are gene expression levels
            y_train: 
        """
        if lam1 is None:
            self.lam1 = self.lam1
            self.lam2 = self.lam2
            self.lam3 = self.lam3
        else:
            self.lam1 = lam1
            self.lam2 = lam2
            self.lam3 = lam3

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_train = torch.tensor(X_train, dtype=torch.float32) # without this line, X_train will still be torch.float64. why?
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        if X_val is not None:
            self.X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        if y_val is not None:
            self.y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            self.y_val = torch.tensor(self.y_val, dtype=torch.float32)

        self.evaluation = evaluation
        self.plot_evaluation = plot_evaluation

        self.id_train = id_train
        self.id_test = id_test

        self.epoch_train = epoch_train
        self.fold_num = fold_num
        
        if require_pretrain_phase:
            print("Pretraining...")
            self._pretrain(X_train, lr=lr_pretrain, epoch_pretrain=epoch_pretrain, batch_size=batch_size)
            print("Pretraining complete.\n")
        if require_train_phase:
            print('Training...')
            self._train(X_train, y_train, lr=lr_train, epoch_train=epoch_train, batch_size=batch_size, 
                        early_stopping=early_stopping, lr_decay=lr_decay, X_val=self.X_val, y_val=self.y_val,
                        c_true_labels_dummy=c_true_labels_dummy)
            print("Training complete.\n")


    def _pretrain(self, X_train, lr, epoch_pretrain, batch_size, optimizer='adam'):
        r"""
        Pretraining phase.
        Train the AE module in scACC and initialize cluster self.. 
        """

        self.init_ae()

        train_loader = torch.utils.data.DataLoader(
            X_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        elif optimizer == '?': # TODO
            NotImplemented

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

        for epoch in range(epoch_pretrain):
            total_loss = 0
            for x in train_loader:
                x = x.to(self.device)
                x_bar, _  = self.ae(x)
                z = self.ae.encoder(x)
                optimizer.zero_grad()
                loss = F.mse_loss(x_bar, x)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                X_bar, _ = self.ae(X_train)
                cur_loss = F.mse_loss(X_bar, X_train)
            print("epoch {}\t\t loss={:.4f}".format(epoch, cur_loss.item()))

            scheduler.step(cur_loss.item())
            for param_group in optimizer.param_groups:
                print("Current learning rate is: {}".format(param_group['lr']))

        print("Initializing cluster centroids...")
        with torch.no_grad():
            z = self.ae.encoder(X_train)
        km = sklearn.cluster.KMeans(n_clusters=self.n_clusters, n_init=20)
        km.fit_predict(z.data.cpu().numpy())
        self.ae.cluster_layer.cluster_centers.data = torch.tensor(km.cluster_centers_).to(self.device)
        print("Done.")


    def _train(self, X_train, y_train, lr, epoch_train, batch_size, early_stopping=False, 
               lr_decay=False, X_val=None, y_val=None,
               c_true_labels_dummy=None):
        r"""
        Training phase.
        """
        # TODO!!!!! sometimes y does not contain all label classes, resulting in bug in get_cluster_entropy 
        train_loader = torch.utils.data.DataLoader(
            subDataset(X_train, y_train), # from utils
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            subDataset(X_val, y_val), # from utils
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)

        # for eval
        epoch_list = []
        trn_loss1_list = []
        trn_loss2_list = []
        trn_loss3_list = []
        val_loss1_list = []
        val_loss2_list = []
        val_loss3_list = []

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # for temp use
        self.cluster_wpheno_mat = []
        # cluster weight
        self.cluster_weight = []
        # cluster entropy
        self.cluster_entropy = []

        # train
        early_stop = False
        if early_stopping:
            early_stop = False
            best_test_loss = float('inf')
            best_train_loss = float('inf')
            patience = 20
            counter = 0

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        w = torch.ones(self.n_clusters, device=get_device()) # NOTE
        w2 = torch.ones(torch.unique(y_train).shape[0], device=get_device()) # NOTE

        w3 = torch.ones(self.n_clusters, device=get_device()) # NOTE 06/16 change
        # w3[0] = 0.2
        # w3[1] = 1.0
        # w3[2] = 0.2

        for epoch in range(epoch_train):
            epoch_list.append(epoch)

            # print(f"Epoch: {epoch+1}: \n --------------------------------- \n")

            if early_stop:
                print("Early stopping triggered!")
                break
            test_loss = 0.0
            train_loss = 0.0
            train_pheno_loss = 0.0
            test_pheno_loss = 0.0

            # NOTE 
            with torch.no_grad():
                z = self.ae.encoder(X_train)
                _, q = self.ae(X_train)
            # p = target_distribution_v2(y_train, q) # NOTE!!!
            p = target_distribution(q) # NOTE!!!

            self.ae.train()
            for x, y, idx in train_loader: 
                # TODO . modify data loader instead of this
                if torch.unique(y).shape[0] != torch.unique(y_train).shape[0]:
                    continue

                # NOTE new version
                x = x.to(self.device)
                x_bar, q = self.ae(x)

                loss1 = F.mse_loss(x_bar, x) # NOTE
                loss2 = F.kl_div(q.log(), p[idx]) # KL-div loss # NOTE do not use reduction='batchmean', it will significantly lower performance (why?)
                # h, _ = get_cluster_entropy(y=y, q=q) # NOTE OLD VERSION
                h, _ = get_cluster_entropy_v3(y=y, q=q, c_true_labels_dummy=c_true_labels_dummy) # NOTE NEW CROSS ENTROPY VERSION
                loss3 = w3.T@h / len(h) # NOTE NEW VERSION
                
                loss = self.lam1*loss1 + self.lam2*loss2 + self.lam3*loss3

                optimizer.zero_grad(set_to_none=True)
                loss.backward() # NOTE!!!!!!!!!!!
                optimizer.step()

                # NOTE old version
                # x = x.to(self.device)
                # x_bar = self.ae(x)
                # z = self.ae.encoder(x)
                # q = calc_q(z, self.clusters.data, self.alpha)
                # p = target_distribution(q.data)
                # loss1 = F.mse_loss(x_bar, x) # AE reconstruction loss
                # loss2 = F.kl_div(q.log(), p) # KL-div loss # NOTE do not use reduction='batchmean', it will significantly lower performance (why?)
                # loss3 = get_pheno_loss(y=y, Q=q, w=w)
                # loss = self.lam2*loss2 + self.lam3*loss3 + loss1
                # train_loss += loss.item()
                # train_pheno_loss += loss3.item()
                # optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                # optimizer.step()

            # # NOTE compute the clusters weights of the current epoch
            # with torch.no_grad():
            #     _, q = self.ae(X_train)
            # h, wpheno = get_cluster_entropy_v2(y=y_train, q=q) # NOTE TEMP
            # # NOTE. enlarge the gap between larger values and smaller values. this is more useful
            # h = torch.pow(h, 2)
            # w = h / h.sum()

            # # TODO ???
            # h, _ = get_phenotype_entropy(y=y_train, q=q) # NOTE TEMP
            # h = torch.pow(h, 2)
            # w2 = h / h.sum()

            # # w = (h - h.min()) / (h.max() - h.min()) + 1e-10 # NOTE NOPE! 0 will distort clustering.
            # # w = h / h.max() # NOTE not strong enough
            # # w += 1e-10; 

            # EVALUATION
            self.ae.eval()
            with torch.no_grad():
                # trn
                x_bar, q = self.ae(X_train)
                p = target_distribution(q)
                loss1 = F.mse_loss(x_bar, X_train)
                trn_loss1_list.append(loss1.item())
                loss2 = F.kl_div(q.log(), p)
                trn_loss2_list.append(loss2.item())
                h, wpheno = get_cluster_entropy_v3(y=y_train, q=q, c_true_labels_dummy=c_true_labels_dummy) # NOTE NEW CROSS ENTROPY VERSION
                trn_loss3_list.append(torch.mean(h).item())
                _, cpheno = get_cluster_entropy_v2(y=y_train, q=q)
                train_ent_loss = torch.mean(h) 
                # val
                x_bar, q = self.ae(X_val)
                p = target_distribution(q)
                loss1 = F.mse_loss(x_bar, X_val)
                val_loss1_list.append(loss1.item())
                loss2 = F.kl_div(q.log(), p)
                val_loss2_list.append(loss2.item())
                h, _ = get_cluster_entropy_v3(y=y_val, q=q, c_true_labels_dummy=c_true_labels_dummy) # NOTE NEW CROSS ENTROPY VERSION
                val_loss3_list.append(torch.mean(h).item())

            # import matplotlib.colors as mcolors
            # cmap = mcolors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (1, 'red')])
            # sns.heatmap(cpheno.cpu().numpy(), cmap=cmap); plt.show()
            clear_output(wait=True)
            _, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            from matplotlib.ticker import MaxNLocator
            axes[0].plot(epoch_list, trn_loss1_list)
            axes[0].plot(epoch_list, val_loss1_list)
            axes[0].legend(['train', 'val'], loc='lower left')
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].set_title('loss1 - rec')
            axes[1].plot(epoch_list, trn_loss2_list)
            axes[1].plot(epoch_list, val_loss2_list)
            axes[1].legend(['train', 'val'], loc='lower left')
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[1].set_title('loss2 - kl div')
            axes[2].plot(epoch_list, trn_loss3_list)
            axes[2].plot(epoch_list, val_loss3_list)
            axes[2].legend(['train', 'val'], loc='lower left')
            axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[2].set_title('loss3 - cluster entropy')
            plt.tight_layout()
            plt.show()

            # LR DECAY
            if lr_decay:
                scheduler.step(train_pheno_loss)
                for param_group in optimizer.param_groups:
                    print("Current learning rate is: {}".format(param_group['lr']))

            if early_stopping:
                if train_ent_loss < best_train_loss:
                    best_train_loss = train_ent_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        early_stop = True
            
            # if early_stopping:
            #     if test_loss < best_val_loss:
            #         best_test_loss = test_loss
            #         counter = 0
            #     else:
            #         counter += 1
            #         if counter >= patience:
            #             early_stop = True
            #             print("Early stopping triggered!")

            # if self.evaluation:
            #     self.evaluate(X_train, y_train, epoch)

    def get_latent_space(self, x):
        r"""
        Get the latent space of input data. 

        Args:
            x [numpy array]: input data

        Returns:
            latent space as a numpy array
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        z = self.ae.encoder(x)
        return z.detach().cpu().numpy()

    def plot_latent_space(self, z, label=None, 
                          title=None, require_distinguishable_colors=False, 
                          s=10, also_plot_cluster=False):

        custom_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FF8000", "#8000FF",
            "#FF007F", "#007FFF", "#7FFF00", "#FF7F00", "#00FF7F", "#7F00FF", "#C0C0C0", "#808080",
            "#400080", "#800040", "#804000", "#008040", "#408000", "#800080", "#408080", "#008080",
            "#804040", "#804080", "#408040", "#800000", "#008000", "#000080"
        ]
        n_colors = 0
        if type(label) == type(torch.Tensor()):
            n_colors = len(np.unique(label))
        elif type(label) == type(pd.DataFrame()):
            n_colors = len(np.unique(label.values))
        elif type(label) == type(pd.Series()):
            n_colors = len(np.unique(label.values.tolist()))
        elif type(label) == type(np.array([])):
            n_colors = len(np.unique(label.tolist()))

        c = self.ae.cluster_layer.cluster_centers.data.cpu().numpy()

        # TODO 
        if require_distinguishable_colors is True:
            sns.scatterplot(x=z[:,0], y=z[:,1], hue=label, s=s,
                    palette=sns.color_palette(custom_colors, n_colors)).set(title=title)
            if also_plot_cluster:
                sns.scatterplot(x=c[:,0], y=c[:,1], s=100, color='black', marker='x')
        else:
            sns.scatterplot(x=z[:,0], y=z[:,1], hue=label, s=s).set(title=title)
            if also_plot_cluster:
                sns.scatterplot(x=c[:,0], y=c[:,1], s=100, color='black', marker='x')
            
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    def get_cluster_embedding(self):
        r"""
        A helper function to get clusters in the embedding
        """
        return self.ae.cluster_layer.cluster_centers.data

    def get_cluster_assignments(self, x):        
        r"""
        A helper function to get point assignments to clusters
        """
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        z = self.ae.encoder(x)
        q, _ = self.ae.cluster_layer(z)
        return assign_cluster(q).detach().cpu().numpy()
    
    def to_device(dev):
        self.ae.to(dev)
        self.device=dev
        self.ae.cluster_layer.to(dev)
        self.ae.device=dev