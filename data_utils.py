import torch, random, os
import numpy as np
from torch_geometric.datasets import QM9, TUDataset, CitationFull
from utils import *
from model import *

# dataset = CitationFull(
#     root='data/Cora',
#     name='Cora_ML'
# )

def make_datasets(
        num_nsamples = 1000,
        num_nclass = 7,
        num_gclass = 5,
        n_feats = 32,
        ng_perclass = 100,
        nn_perclass = (50, 100),
        nlabel_perm = 0.5,
        graph_selec_noise = 0.5,
        cov_scale = 2,
        train_per = 0.85,
        test_per = 0.15,
        norm_mode = "max",
        visualize = False
        ):
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(num_gclass)])

    s_dataset = RandomGraphDatset(
        num_gclass,
        num_nclass,
        n_feats,
        cov_scale,
        main_ds=True
        )
    s_dataset.gen_xy(
        num_nsamples=num_nsamples,
        nlabel_perm=nlabel_perm,
        normalize_mode=norm_mode
        )
    s_dataset.gen_graph_ds(
        num_nsamples,
        nlabel_perm,
        ng_perclass,
        nn_perclass,
        nnoise_selec_std=graph_selec_noise,
        ngclass_probs=None
        )
    s_dataset.to_tensor()
    s_dataset.init_loaders(train_per=train_per, test_per=test_per)

    t_dataset = RandomGraphDatset(
        num_gclass,
        num_nclass,
        init_gmm=s_dataset.gmm,
        main_ds=False
        )
    t_dataset.gen_xy(
        num_nsamples=num_nsamples,
        nlabel_perm=nlabel_perm,
        normalize_mode=None
        )
    mean = np.zeros(t_dataset.x.shape[1])
    cov_matrix = np.eye(t_dataset.x.shape[1]) * cov_scale
    t_dataset.add_multivariate_noise(mean, cov_matrix)
    t_dataset.normalize_feats(mode=norm_mode)
    t_dataset.gen_graph_ds(
        num_nsamples,
        nlabel_perm,
        ng_perclass,
        nn_perclass,
        nnoise_selec_std=graph_selec_noise,
        ngclass_probs=s_dataset.ngclass_probs
        )
    t_dataset.to_tensor()
    t_dataset.init_loaders(train_per=train_per, test_per=test_per)

    if visualize:
        s_dataset.visualize("/content/CPrompt/tsne_source.png")
        t_dataset.visualize("/content/CPrompt/tsne_destination.png")
    
    return s_dataset, t_dataset

def get_dataset(
        ds_name,
        cov_scale = 2,
        train_per = 0.85,
        test_per = 0.15,
        norm_mode = "max",
        node_attributes = True,
        visualize = False):
    
    """ Currently supported datasets: 
        - ENZYMES
        - PROTEINS_full
    """
    dataset = TUDataset(
        root = 'data/TUDataset',
        name = ds_name,
        use_node_attr = node_attributes
        )

    ntotal_graphs = len(dataset)
    perm = torch.randperm(ntotal_graphs)
    s_perm = perm[:int(ntotal_graphs*0.5)]
    t_perm = perm[int(ntotal_graphs*0.5):]
    s_ds = dataset.copy(s_perm)
    t_ds = dataset.copy(t_perm)

    s_dataset = ToGraphDataset(s_ds, normalize=True, normalize_mode="max")
    s_dataset.gen_graph_ds(s_ds)
    s_dataset.init_loaders(train_per=train_per, test_per=test_per, shuffle=False)
    t_dataset = ToGraphDataset(t_ds)
    mean = np.zeros(t_dataset.n_feats)
    cov_matrix = np.eye(t_dataset.n_feats) * cov_scale
    t_dataset.add_multivariate_noise(mean, cov_matrix)
    t_dataset.normalize_feats(normalize_mode=norm_mode)
    t_dataset.gen_graph_ds(t_ds)
    t_dataset.init_loaders(train_per=train_per, test_per=test_per)

    if visualize:
        s_dataset.visualize("/content/CPrompt/tsne_source.png")
        t_dataset.visualize("/content/CPrompt/tsne_destination.png")

    return s_dataset, t_dataset