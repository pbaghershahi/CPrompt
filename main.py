
from train_utils import *
from utils import *
from data_utils import *
import argparse

def main(args):
    s_dataset, t_dataset = get_node_dataset(
        "Cora",
        cov_scale = 0.1,
        mean_shift = 0.,
        train_per = 0.80,
        test_per = 0.20,
        batch_size = 32,
        n_hopes = 2,
        norm_mode = "max",
        node_attributes = True,
        visualize = False
    )

    model_name = "GCN"
    model_config = dict(
        d_feat = s_dataset.n_feats, 
        d_hid = 64, 
        d_class = s_dataset.num_gclass, 
        n_layers = 2, r_dropout = 0.2
    )
    optimizer_config = dict(
        lr = 1e-2,
        scheduler_step_size = 100,
        scheduler_gamma = 0.5
    )
    training_config = dict(
        n_epochs = 50
    )

    pretrained_path = pretrain_model(
        s_dataset,
        model_name, 
        model_config,
        optimizer_config,
        training_config,
        eval_step = 1,
        save_model = True, 
        pretext_task = "classification",
        model_dir = "./pretrained"
    )

    pretrained_config = model_config
    optimizer_config = dict(
        lr = 1e-3,
        scheduler_step_size = 100,
        scheduler_gamma = 1.0
    )
    if args.prompt_method == "all_in_one":
        prompt_config = dict(
            token_dim = t_dataset.n_feats,
            token_num = 10, 
            cross_prune = 0.1, 
            inner_prune = 0.3, 
            trans_x = False
        )
        training_config = dict(
            n_epochs = 150
        )
    elif args.prompt_method == "contrastive":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = 64,
            output_dim = t_dataset.n_feats,
            prompt_fn = "add_tokens",
            token_num = 10
        )
        training_config = dict(
            aug_type = "feature",
            pos_aug_mode = "mask",
            neg_aug_mode = "arbitrary",
            p_raug = 0.15,
            n_raug = 0.15,
            add_link_loss = False,
            n_epochs = 150,
        )
    elif args.prompt_method == "pseudo_labeling":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = 64,
            output_dim = t_dataset.n_feats,
            prompt_fn = "add_tokens",
            token_num = 10
        )
        training_config = dict(
            aug_type = "feature",
            pos_aug_mode = "mask",
            p_raug = 0.15,
            n_epochs = 150,
        )
    else:
        raise Exception("The chosen method is not valid!")
    
    prompting(
            t_dataset,
            args.prompt_method, 
            prompt_config,
            pretrained_config,
            optimizer_config,
            pretrained_path,
            training_config,
            s_dataset,
            num_runs = 5,
            eval_step = 1
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAttLE')
    parser.add_argument("-m", "--prompt-method", type=str, required=True,
                        help="all_in_one, contrastive, pseudo_labeling")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=1,
                        help="learning rate decay rate")

    args = parser.parse_args()
    main(args)