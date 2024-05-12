from train_utils import *
from utils import *
from data_utils import *
import argparse

def main(args):
    os.makedirs('./log', exist_ok=True)
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    t_ds_name = args.t_dataset if args.t_dataset != args.s_dataset else args.s_dataset
    logger.info(f"Source Dataset: {args.s_dataset}, Target Dataset: {t_ds_name}. Training, Test: {args.train_per}, {args.test_per} -- Batch size: {args.batch_size}")

    if args.s_dataset in ["Cora", "CiteSeer", "PubMed"]:
        s_dataset, t_dataset = get_node_dataset(
            args.s_dataset,
            cov_scale = 0.2,
            mean_shift = 0.,
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
            cov_scale = 0.2,
            mean_shift = 0.,
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
        d_feat = s_dataset.n_feats, 
        d_hid = args.pretrain_h_dim, 
        d_class = s_dataset.num_gclass, 
        n_layers = 2, r_dropout = 0.2
    )
    optimizer_config = dict(
        lr = args.pretrain_lr,
        scheduler_step_size = args.pretrain_step_size,
        scheduler_gamma = args.pretrain_lr
    )
    training_config = dict(
        n_epochs = args.pretrain_n_epochs
    )
    logger.info(f"Setting for pretraining: Model: {model_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")

    if args.pretrain:
        logger.info(f"Pretraining {model_name} on {args.s_dataset} started for {args.pretrain_n_epochs} epochs")
        _, pretrained_path = pretrain_model(
        s_dataset,
        # t_dataset,
        model_name, 
        model_config,
        optimizer_config,
        training_config,
        logger,
        eval_step = args.pretrain_eval_step,
        save_model = args.save_pretrained, 
        pretext_task = "classification",
        model_dir = "./pretrained",
        empty_pretrained_dir = args.empty_pretrained_dir
        )
        logger.info(f"Pretraining is finished! Saved to: {pretrained_path}")
    else:
        pretrained_path = args.pretrain_path
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