import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils import *
from data_utils import *
from prompt_func import *
from model import *

def pretrain_model(
    s_dataset,
    model_name, 
    model_config,
    optimizer_config,
    training_config,
    logger,
    eval_step = 1,
    save_model = True, 
    pretext_task = "classification",
    model_dir = "./pretrained",
    empty_pretrained_dir = False
):
    task = "multi" if s_dataset.num_gclass > 2 else "binary"
    if model_name == "GCN":
        model = PretrainedModel(**model_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if pretext_task == "classification":
        obj_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    test_loss, test_acc, test_f1 = test(model, s_dataset, device, task = task, mode = "pretrain", validation = False)
    logger.info(f'GNN Before Pretraining -- Test Loss: {test_loss:.4f} -- Test ACC: {test_acc:.3f} -- Test F1-score: {test_f1:.3f}')

    n_epochs = training_config["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
        for i, batch in enumerate(s_dataset.train_loader):
            optimizer.zero_grad()
            scores, _ = model(
                batch,
                decoder = True,
                device = device
            )
            # ipdb.set_trace()
            loss = obj_fun(scores, batch.y.to(device))
            loss.backward()
            optimizer.step()
            # if i % max(1, int((s_dataset.n_train//scores.size(0))*0.5)) == 0:
            #     logger.info(f"Train batch: {i}/{np.ceil(s_dataset.n_train//scores.size(0))} -- Train Loss: {loss.item()}")
        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0:
            valid_loss, valid_acc, valid_f1 = test(model, s_dataset, device, task = task, mode = "pretrain", validation = True)
            logger.info(
                f"Epoch: {epoch}/{n_epochs} -- Train Loss: {loss:.4f} -- " +
                f"Validation Loss: {valid_loss:.4f} -- Validation ACC: {valid_acc:.3f} -- Validation F1: {valid_f1:.3f}"
            )

    test_loss, test_acc, test_f1 = test(model, s_dataset, device, task = task, mode = "pretrain", validation = False)
    logger.info(
        "#"*10 + " " +
        f"Final Results: -- Train Loss: {loss:.4f} -- " +
        f"Test Loss: {test_loss:.4f} -- Test ACC: {test_acc:.3f} -- Test F1: {test_f1:.3f}" +
        " " + "#"*10
    )
    if empty_pretrained_dir:
        empty_directory(model_dir)
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        if isinstance(logger, logging.Logger):
            exec_name = logger.handlers[1].baseFilename.split("/")[-1].split(".log")[0]
        else:
            exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        model_path = os.path.join(model_dir, f"{model_name}_Pretrained_{exec_name}.pth")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path
        )
    else:
        model_path = "Won't be stored"
    return model, model_path
    

def prompting(
    t_dataset,
    prompt_method, 
    prompt_config,
    pretrained_config,
    optimizer_config,
    pretrained_path,
    training_config,
    logger,
    s_dataset = None,
    num_runs = 5,
    eval_step = 1
):
    task = "multi" if t_dataset.num_gclass > 2 else "binary"
    overal_test_acc = []
    overal_valid_acc = []
    overal_test_f1 = []
    overal_valid_f1 = []
    for k in range(num_runs):
    
        main_model = PretrainedModel(**pretrained_config)
        if prompt_method == "all_in_one":
            pmodel = HeavyPrompt(**prompt_config)
        elif prompt_method == "contrastive":
            pmodel = BasePrompt(**prompt_config)
        elif prompt_method == "pseudo_labeling":
            pmodel = BasePrompt(**prompt_config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main_model.to(device)
        pmodel.to(device)
        load_model(main_model, read_checkpoint=True, pretrained_path=pretrained_path)
        for param in main_model.parameters():
            param.requires_grad = False
        main_model.eval()
    
        optimizer = Adam(pmodel.parameters(), lr = optimizer_config["lr"])
        scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
        Trainer = PromptTrainer(prompt_method, training_config)
    
        if s_dataset is not None:
            test_loss, test_acc, test_f1 = test(main_model, s_dataset, device, task = task, mode = "pretrain", validation = False)
            logger.info(f'Pretrained GNN on Source Dataset -- Test Loss: {test_loss:.4f} -- Test ACC: {test_acc:.3f} -- Test F1-score: {test_f1:.3f}')
        test_loss, test_acc, test_f1 = test(main_model, t_dataset, device, task = task, mode = "pretrain", validation = False)
        logger.info(f'Pretrained GNN on Target Dataset Without Prompting -- Test Loss: {test_loss:.4f} -- Test ACC: {test_acc:.3f} -- Test F1-score: {test_f1:.3f}')
    
        valid_average_acc = []
        valid_average_f1 = []
        n_epochs = training_config["n_epochs"]
        for epoch in range(n_epochs):
            pmodel.train()
            main_model.eval()
            Trainer.train(t_dataset, main_model, pmodel, optimizer, device, logger)
            scheduler.step()
            optimizer.zero_grad()
            
            if epoch % eval_step == 0 or epoch >= n_epochs - 6:
                pmodel.eval()
                main_model.eval()
                valid_loss, valid_acc, valid_f1 = test(main_model, t_dataset, device, task = task, mode = "prompt", pmodel = pmodel, validation = True)
                logger.info(f"Epoch: {epoch}/{n_epochs} -- Validation Loss: {valid_loss:.4f} -- Validation ACC: {valid_acc:.3f} -- Validation F1: {valid_f1:.3f}")
                if epoch >= n_epochs - 6:
                    valid_average_acc.append(valid_acc)
                    valid_average_f1.append(valid_f1)
        n_evali_valid = len(valid_average_f1)
        valid_average_acc = np.array(valid_average_acc).mean()
        valid_average_f1 = np.array(valid_average_f1).mean()
        logger.info(f"Run {k}/{num_runs}: Average Over Last {n_evali_valid} epochs -- Valid ACC: {valid_average_acc} -- Valid F1-score: {valid_average_f1}")
        overal_valid_acc.append(valid_average_acc)
        overal_valid_f1.append(valid_average_f1)

        test_loss, test_acc, test_f1 = test(main_model, t_dataset, device, task = task, mode = "prompt", pmodel = pmodel, validation = False)
        logger.info(f"Test Results of Run {k}/{num_runs}: -- Test Loss: {test_loss:.4f} -- Test ACC: {test_acc:.3f} -- Test F1: {test_f1:.3f}")
        overal_test_acc.append(test_acc)
        overal_test_f1.append(test_f1)
        
    overal_valid_acc = np.array(overal_valid_acc).mean()
    overal_valid_f1 = np.array(overal_valid_f1).mean()
    overal_test_acc = np.array(overal_test_acc).mean()
    overal_test_f1 = np.array(overal_test_f1).mean()
    logger.info(f"Validation average after {num_runs} runs -- ACC: {np.array(overal_valid_acc).mean()} -- F1-score: {np.array(overal_valid_f1).mean()}")
    logger.info(f"Test average after {num_runs} runs -- ACC: {np.array(overal_test_acc).mean()} -- F1-score: {np.array(overal_test_f1).mean()}")
    return pmodel