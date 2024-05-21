import argparse, time, logging, torch, yaml, os, ipdb
from datetime import datetime
from utils import *
from data_utils import *
from train_utils import *
from prompt_func import *
from model import *
import ray
from ray import tune, train
from ray.train import Checkpoint, RunConfig
from ray.util.check_serialize import _Printer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from functools import partial


def file_name_creator(trial,):
    return datetime.today().strftime('%Y-%m-%d-%H-%M')+'-fb15k-237-trail'

def dir_name_creator(trial,):
    return datetime.today().strftime('%Y-%m-%d-%H-%M')+'-fb15k-237-dir'

def train_pretrain(config, args, dataset, logger):

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_config = dict(
        gnn_type = args.gnn_type,
        in_channels = dataset.n_feats,
        hidden_channels = config["gnn_h_dim"],
        out_channels = dataset.num_gclass,
        num_layers = args.gnn_num_layers, 
        dropout = config["gnn_dropout"],
        with_bn = False,
        with_head = True,
    )
    optimizer_config = dict(
        lr = config["gnn_lr"],
        scheduler_step_size = config["gnn_step_size"],
        scheduler_gamma = config["gnn_gamma"]
    )
    training_config = dict(
        n_epochs = args.gnn_n_epochs
    )
    logger.info(f"Setting for pretraining: Model: {model_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")

    if args.pretrain:
        logger.info(f"Pretraining {args.gnn_type.upper()} on {args.s_dataset} started for {args.gnn_n_epochs} epochs")
        _, pretrained_path = pretrain_model(
            dataset,
            args.gnn_type.upper(),
            model_config,
            optimizer_config,
            training_config,
            logger,
            eval_step = args.gnn_eval_step,
            save_model = args.save_pretrained,
            pretext_task = "classification",
            model_dir = "./pretrained",
            empty_pretrained_dir = args.empty_pretrained_dir,
            tunning = True
        )
        logger.info(f"Pretraining is finished! Saved to: {pretrained_path}")
    else:
        pretrained_path = args.gnn_path
        logger.info(f"Using previous pretrained model at {pretrained_path}")


def main(
        args, dataset, logger, 
        num_samples, max_num_epochs
    ):

    config = {
        "gnn_h_dim": tune.choice([64, 128, 256, 512]),
        'gnn_num_layers': tune.choice([2, 3]),
        "gnn_dropout": tune.choice([0.2, 0.3, 0.4, 0.5]),
        'gnn_lr': tune.choice([1e-3]),
        'gnn_step_size': tune.choice([100]),
        "gnn_gamma": tune.choice([0.5, 1])
    }

    search_alg = BasicVariantGenerator(
        # points_to_evaluate=[{
        #     'dr_input': 0.2,
        #     'dr_hid1': 0.2,
        #     'dr_hid2': 0.2,
        #     'dr_output': 0.2,
        #     "dr_decoder": 0.2
        # }],
        max_concurrent=3
    )

    scheduler = ASHAScheduler(
        max_t = max_num_epochs,
        grace_period = 10,
        reduction_factor = 2
        # rungs are calculated based on grace_period and reduction factor as follows:
        # rung_level = grace_period*reduction_factor**i for i in 0, 1, ...
        # for example when grace_period = 2 and reduction_factor = 3, it means that all dat
        # are present for first 2 epochs, then 2/3 of data drops and only the remaining 1/3
        # promots to 6 epochs, then again only 1/3 of data promots to next rung which end
        # after epoch 18 and so on.
    )

    ray.init(
        _temp_dir = "/workspace/tune"
        # logging_level="INFO",  # Set the logging level to INFO (or your preferred level)
        # log_to_driver=True,     # Log to the driver process
        # logging_format="%(asctime)s [%(levelname)s] %(message)s",  # Optional: Customize the log format
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_pretrain, args = args, dataset = dataset, logger = logger),
            resources={"gpu": 1}
        ),
        tune_config = tune.TuneConfig(
            search_alg = search_alg,
            metric = "f1-score",
            mode = "max",
            scheduler = scheduler,
            num_samples = num_samples,
        ),
        param_space = config,
        run_config = RunConfig(
            storage_path = "/workspace/tune",
            log_to_file = True,
            # log_to_file = "/content/TGCN/data/std_combined.log"
        )
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric = "f1-score", mode = "max", scope = "last")
    print(f"Best trial config: Tunned params: {best_result.config} -- Default params: {args}")
    print(f"Best trial saved to: {best_result.path}")
    print(f"Best trial final validation: {best_result.metrics}")
    result_df = best_result.metrics_dataframe
    result_df = result_df.drop(columns=[
        "timestamp", "checkpoint_dir_name", "date", 
        "time_this_iter_s", "time_total_s", "hostname",
        "node_ip", "time_since_restore", "iterations_since_restore",
    ])
    os.makedirs(f"/cpromptdata/tuned/{args.s_dataset}", exist_ok=True)
    result_df.to_csv(f"/cpromptdata/tuned/{args.s_dataset}/tuned_metrics.csv", index=False, sep=',', mode='a')
    print("Metrics of all the trials.")
    print(result_df)


if __name__ == '__main__':

    args = argparse.Namespace(
        pretrain = True,
        # prompt_method = "contrastive",
        prompt_method = "pseudo_labeling",
        # prompt_method = "all_in_one",
        prompt_fn = "add_tokens",
        # dataset = "ego_network",
        s_dataset = "ENZYMES",
        t_dataset = "ENZYMES",
        # s_dataset = "PROTEINS_full",
        # t_dataset = "PROTEINS_full",
        # s_dataset = "Cora",
        # t_dataset = "Cora",
        # s_dataset = "CiteSeer",
        # t_dataset = "CiteSeer",
        # s_dataset = "digg",
        # t_dataset = "oag",
        # s_dataset = "Letter-low",
        # t_dataset = "Letter-high",
        gnn_type = "gcn",
        gnn_num_layers = 2,
        gnn_path = "./pretrained/GCN_Pretrained_2024-05-17-19-19-13.pth",
        gnn_n_epochs = 300,
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
        config_to_file = "./config/enzymes.yaml"
    )

    os.makedirs('./log', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    logger = DummyLogger()
    t_ds_name = args.t_dataset if args.t_dataset != args.s_dataset else args.s_dataset
    logger.info(f"All args: {args}")
    logger.info(
        f"Source Dataset: {args.s_dataset}, Target Dataset: {t_ds_name}. " + 
        f"Training, Test: {args.train_per}, {args.test_per} -- Batch size: {args.batch_size}"
    )
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


    main(
        args, s_dataset, logger, 
        num_samples = 60, max_num_epochs = 20,
    )