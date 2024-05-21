from train_utils import *
from utils import *
from data_utils import *
import yaml
import argparse

def main(args) -> None:
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    t_ds_name = args.t_dataset if args.t_dataset != args.s_dataset else args.s_dataset
    logger.info(f"Source Dataset: {args.s_dataset}, Target Dataset: {t_ds_name}. Training, Test: {args.train_per}, {args.test_per} -- Batch size: {args.batch_size}")
    if args.config_from_file == "":
        if args.config_to_file != "":
            with open(args.config_to_file, 'w') as outfile:
                yaml.dump(vars(args), outfile, default_flow_style=False)
            logger.info(f"Config saved to file: {args.config_to_file}")
    else:
        logger.info(f"Reading config from: {args.config_from_file}")
        with open(args.config_from_file, 'r') as infile:
            args = yaml.safe_load(infile)
            args = argparse.Namespace(**args)
            
    if args.s_dataset in ["Cora", "CiteSeer", "PubMed"]:
        s_dataset, t_dataset = get_node_dataset(
            args.s_dataset,
            cov_scale = args.noise_cov_scale,
            mean_shift = args.noise_mean_shift,
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            n_hopes = 2,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed
        )
    elif args.s_dataset in ["ENZYMES", "PROTEINS_full"]:
        s_dataset, t_dataset = get_graph_dataset(
            args.s_dataset,
            cov_scale = args.noise_cov_scale,
            mean_shift = args.noise_mean_shift,
            store_to_path = "./data/TUDataset",
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed
        )
    elif args.s_dataset in ["Letter-high", "Letter-low", "Letter-med"]:
        s_dataset, t_dataset = get_pyggda_dataset(
            args.s_dataset,
            t_ds_name,
            store_to_path = "./data/TUDataset",
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed
        )
    elif args.s_dataset in ["digg", "oag", "twitter", "weibo"]:
        s_dataset, t_dataset = get_gda_dataset(
            ds_dir = "./data/ego_network/",
            s_ds_name = args.s_dataset,
            t_ds_name = t_ds_name,
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            get_s_dataset = True,
            get_t_dataset = True,
            seed = args.seed
        )

    model_name = "GCN"
    model_config = dict(
        gnn_type = args.gnn_type,
        in_channels = s_dataset.n_feats,
        hidden_channels = args.gnn_h_dim,
        out_channels = s_dataset.num_gclass,
        num_layers = args.gnn_num_layers, 
        dropout = args.gnn_dropout,
        with_bn = False,
        with_head = True,
    )
    optimizer_config = dict(
        lr = args.gnn_lr,
        scheduler_step_size = args.gnn_step_size,
        scheduler_gamma = args.gnn_gamma
    )
    training_config = dict(
        n_epochs = args.gnn_n_epochs
    )
    logger.info(f"Setting for pretraining: Model: {model_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")

    if args.pretrain:
        logger.info(f"Pretraining {model_name} on {args.s_dataset} started for {args.gnn_n_epochs} epochs")
        _, pretrained_path = pretrain_model(
            s_dataset,
            model_name,
            model_config,
            optimizer_config,
            training_config,
            logger,
            eval_step = args.gnn_eval_step,
            save_model = args.save_pretrained,
            pretext_task = "classification",
            model_dir = "./pretrained",
            empty_pretrained_dir = args.empty_pretrained_dir
        )
        logger.info(f"Pretraining is finished! Saved to: {pretrained_path}")
    else:
        pretrained_path = args.gnn_path
        logger.info(f"Using previous pretrained model at {pretrained_path}")

    pretrained_config = model_config
    num_tokens = int(np.ceil(cal_avg_num_nodes(t_dataset)))
    logger.info(f"The number of tokens added: {num_tokens}")
    optimizer_config = dict(
        lr = args.lr,
        scheduler_step_size = args.step_size,
        scheduler_gamma = args.gamma
    )
    logger.info(f"Prompting method: {args.prompt_method} -- Setting: Prompting function: {args.prompt_fn} -- Target Dataset: {t_ds_name}")
    if args.prompt_method == "all_in_one":
        prompt_config = dict(
            token_dim = t_dataset.n_feats,
            token_num = num_tokens,
            cross_prune = args.cross_prune,
            inner_prune = args.inner_prune,
            trans_x = args.trans_x
        )
        training_config = dict(
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Transform X: {args.trans_x} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    elif args.prompt_method == "contrastive":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = args.h_dim,
            output_dim = t_dataset.n_feats,
            prompt_fn = args.prompt_fn,
            token_num = num_tokens
        )
        training_config = dict(
            aug_type = args.aug_type,
            pos_aug_mode = args.pos_aug_mode,
            neg_aug_mode = args.neg_aug_mode,
            p_raug = args.p_raug,
            n_raug = args.n_raug,
            add_link_loss = args.add_link_loss,
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Augmentation type: {args.aug_type} -- Positive aug mode and rate: {args.pos_aug_mode}, {args.p_raug} -- Negative aug mode and rate: {args.neg_aug_mode}, {args.n_raug} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    elif args.prompt_method == "pseudo_labeling":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = args.h_dim,
            output_dim = t_dataset.n_feats,
            prompt_fn = args.prompt_fn,
            token_num = num_tokens
        )
        training_config = dict(
            aug_type = args.aug_type,
            pos_aug_mode = args.pos_aug_mode,
            p_raug = args.p_raug,
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Augmentation type: {args.aug_type} -- Positive aug mode and rate: {args.pos_aug_mode}, {args.p_raug} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    else:
        raise Exception("The chosen method is not valid!")

    logger.info(f"Setting for prompt tuning: Prompt: {prompt_config} -- Pretrained Model: {pretrained_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")
    logger.info(f"Prompt tuning started: Num runs: {args.num_runs} -- Eval step: {args.eval_step}")
    pmodel = prompting(
        t_dataset,
        args.prompt_method,
        prompt_config,
        pretrained_config,
        optimizer_config,
        pretrained_path,
        training_config,
        logger,
        s_dataset,
        num_runs = args.num_runs,
        eval_step = args.eval_step
    )


if __name__ == "__main__":
    args = argparse.Namespace(
        pretrain = True,
        # prompt_method = "contrastive",
        prompt_method = "pseudo_labeling",
        # prompt_method = "all_in_one",
        prompt_fn = "add_tokens",
        # dataset = "ego_network",
        # s_dataset = "ENZYMES",
        # t_dataset = "ENZYMES",
        # s_dataset = "PROTEINS_full",
        # t_dataset = "PROTEINS_full",
        s_dataset = "Cora",
        t_dataset = "Cora",
        # s_dataset = "CiteSeer",
        # t_dataset = "CiteSeer",
        # s_dataset = "digg",
        # t_dataset = "oag",
        # s_dataset = "Letter-low",
        # t_dataset = "Letter-high",
        gnn_type = "gcn",
        gnn_num_layers = 2,
        gnn_path = "./pretrained/GCN_Pretrained_2024-05-17-19-19-13.pth",
        gnn_n_epochs = 150,
        gnn_eval_step = 10,
        gnn_h_dim = 256,
        gnn_lr = 1e-3,
        gnn_step_size = 100,
        gnn_gamma = 0.5,
        gnn_batch_size = 32,
        gnn_dropout = 0.2,
        save_pretrained = True,
        batch_size = 32,
        lr = 1e-3,
        step_size = 100,
        gamma = 0.75,
        aug_type = "feature",
        pos_aug_mode = "mask",
        neg_aug_mode = "arbitrary",
        h_dim = 256,
        add_link_loss = False,
        empty_pretrained_dir = False,
        noise_cov_scale = 2,
        noise_mean_shift = 2,
        cross_prune = 0.1,
        inner_prune = 0.3,
        trans_x = False,
        n_epochs = 200,
        num_runs = 5,
        eval_step = 5,
        p_raug = 0.15,
        n_raug = 0.15,
        r_reg = 0.2,
        train_per = 0.7,
        test_per = 0.2,
        seed = 4321,
        config_from_file = "",
        config_to_file = "./config/cora.yaml"
    )
    main(args)