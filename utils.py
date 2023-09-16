from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset
import warnings
import pandas as pd
from scipy.stats import entropy
warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = 'cuda'
def set_device(device_set):
    global device 
    device = device_set
def get_device():
    global device
    return device

def assign_cluster(q):
    r"""
    Assign cells to clusters based on softmax.
    
    Args:
        matrix q where q_{ij} measures the probability that embedded point z_i
        belongs to centroid j. (q.shape == [num_cells, num_clusters])
    Returns:
        assigned clusters of cells (assigns.shape == [num_cells,])
    """
    assigns = torch.max(q, 1)[1]
    return assigns

def pairwise_dist(q1, q2, p=2):
    """
    pairwise distance in the z space[[based on q of the clusters]]
    """
    # return torch.cdist(q1, q2, p=p)
    return pearsonr(q1, q2)


def get_phenotype_entropy(y, q):
    r"""
    
    Returns:
        tesor with shape (c,1): entropy of each phenotype across clusters
    """
    y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    wpheno = q.T@y + 1e-15 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
    wpheno /= wpheno.sum(dim=0, keepdim=True)          
    h = -1*torch.sum(wpheno*wpheno.log(), 0)
    return h, wpheno

def get_cluster_entropy(y, q):
    r"""
    
    Returns:
        tesor with shape (c,1): entropy of each cluster
    """
    # OLD VERSION
    y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    wpheno = q.T@y + 1e-15 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
    wpheno /= wpheno.sum(dim=1, keepdim=True)          
    h = -1*torch.sum(wpheno*wpheno.log(), 1)
    return h, wpheno

    # # NEW VERSION: replace q in the old version which is soft probabilities with hard assignments
    # #               ??? will this be better? in old version wpheno will be much bigger than cpheno in new version,
    # #                                       so the entropy will be smaller in new version                              
    # y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    # assigns = torch.where(q == q.max(dim=1, keepdim=True).values, 1.0, 0.0)
    # cpheno = assigns.T@y + 1e-15 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
    # cpheno /= cpheno.sum(dim=1, keepdim=True)          
    # h = -1*torch.sum(cpheno*cpheno.log(), 1)
    # return h, cpheno

def get_cluster_entropy_v2(y, q):
    # NEW VERSION: replace q in the old version which is soft probabilities with hard assignments
    #               ??? will this be better? in old version wpheno will be much bigger than cpheno in new version,
    #                                       so the entropy will be smaller in new version                              
    y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    assigns = torch.where(q == q.max(dim=1, keepdim=True).values, 1.0, 0.0)
    cpheno = assigns.T@y + 1e-15 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
    cpheno /= cpheno.sum(dim=1, keepdim=True)          
    h = -1*torch.sum(cpheno*cpheno.log(), 1)
    return h, cpheno

def get_cluster_entropy_v3(y, q, c_true_labels_dummy):
    y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    wpheno = q.T@y + 1e-15 # weighted phenotype by clusters. add 1e-9 to prevent log(0)
    wpheno /= wpheno.sum(dim=1, keepdim=True) 
    h = -1*torch.sum(c_true_labels_dummy*wpheno.log(), 1)
    return h, wpheno

def target_distribution(q):
    r"""
    Computes and returns the target distribution P based on Q.
    
    Args:
        q: similarity between embedded point z_i and cluster center j 
            measured by Student's t-distribution
    Returns:
        a tensor (matrix) where the (i,j) element is p_{ij}
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def target_distribution_v2(y, q):
    # NEW VERSION: p = y(y y.T)y.T q
    r"""
    Computes and returns the target distribution P based on Q.
    
    Args:
        q: similarity between embedded point z_i and cluster center j 
            measured by Student's t-distribution
    Returns:
        a tensor (matrix) where the (i,j) element is p_{ij}
    """
    # old version: gpu memory not enough
    # y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    # Y = y@y.T
    # Y = Y / Y.sum(axis=1)
    # p = Y@q

    # new version: save memory
    temp = torch.unique(y, return_counts=True)
    dic = dict(zip(temp[0].tolist(), temp[1].tolist()))        
    tempp = [dic[i.item()] for i in y]
    denominator_temp = torch.tensor(tempp, device=get_device(), dtype=torch.float32).reshape(-1,1)
    y = torch.tensor(pd.get_dummies(y.cpu().numpy()).to_numpy(), device=get_device(), dtype=torch.float32)
    p = (y@(y.T@q)) / denominator_temp
    return p

def calc_entropy(q, y): 
    r"""
    Ags:
        q: 
        y: 
    """
    assigns = assign_cluster(q)
    centroids = torch.unique(assigns) # assigned centroids
    ent = 0
    for centroid in centroids:
        counts = torch.unique(y[assigns==centroid], return_counts=True)[1]
        p = counts / torch.sum(counts)
        ent += torch.sum(-p*torch.log(p))
    return ent / centroids.shape[0]

def calc_q(z, cluster_centroids, alpha=1):
    r"""
    Compute Q (q_{ij} matrix) matrix from embedding data.
    
    Args:
        z: mapping of input x into the hidden layer
        cluster_centroids: cluster centroids in latent space
    
    Returns: 
        Soft assignment probability matrix of point i to cluster j.
    
    NOTES:
        It is order (row-wise) preserving.
    """
    q = 1.0 / \
        (1.0 + \
            ( (torch.sum(torch.pow(z.unsqueeze(1) - cluster_centroids, 2), 2)) \
                / alpha ) )
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

class subDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        """
        Returns the number of samples in the dataset. 
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index idx.
        """
        return self.x[idx], \
               self.y[idx], \
               torch.tensor(idx).to(torch.int64)

def calc_q_chunks(z, cluster_centers, alpha=1): # to save memory. CONFIRMED, CORRECT IMPLEMENTATION.
    q_total = []

    for z_chunk in z.chunk(10):  # change the number of chunks as needed
        dists = torch.sum(torch.pow(z_chunk.unsqueeze(1) - cluster_centers, 2), 2)
        q_chunk = 1.0 / (1.0 + (dists / alpha))
        q_chunk = q_chunk.pow((alpha + 1.0) / 2.0)
        q_chunk = (q_chunk.t() / torch.sum(q_chunk, 1)).t()
        q_total.append(q_chunk)

    q = torch.cat(q_total)
    return q
