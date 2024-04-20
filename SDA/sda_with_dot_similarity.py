def entropy_loss(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

average_acc = []
for _ in range(5):

    h_dim = 64
    ph_dim = 64
    o_dim = 64
    n_layers = 2
    n_epochs = 150
    temperature = 1
    n_drops = 0.15
    batch_size = 32
    n_augs = 2
    aug_type = "feature"
    aug_mode = "mask"
    add_link_loss = False
    visualize = False
    if visualize:
        colors = np.array([
            "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
            for i in range(s_dataset.y.unique().size(0))])
    else:
        colors = None


    # seed_value = 27324
    # torch.manual_seed(seed_value)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed_value)

    enc_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
    main_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pmodel = LinkPredictionPrompt(
        t_dataset.n_feats,
        h_dim,
        t_dataset.n_feats,
        prompt_fn = "add_tokens",
        token_num = 30,
        device="cuda:0"
    )
    # pmodel = LinkPredictionPrompt(
    #     t_dataset.n_feats,
    #     h_dim, t_dataset.n_feats,
    #     num_layers = 2,
    #     normalize = True,
    #     has_head = False,
    #     device = device,
    #     has_gnn_encoder = False
    # )
    enc_model.to(device)
    main_model.to(device)
    pmodel.to(device)
    obj_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pmodel.parameters(), lr=1e-3)
    pfiles_path = [pfile for pfile in os.listdir("/content/CPrompt/pretrained/") if pfile.endswith(".pt")]
    prepath = os.path.join("/content/CPrompt/pretrained/", pfiles_path[0])
    load_model(enc_model, read_checkpoint=True, pretrained_path=prepath)
    load_model(main_model, read_checkpoint=True, pretrained_path=prepath)
    for param in enc_model.parameters():
        param.requires_grad = False
    for param in main_model.parameters():
        param.requires_grad = False

    ug_graphs = []
    losses = []
    main_losses = []
    main_accs = []
    enc_model.eval()
    main_model.eval()

    with torch.no_grad():
        main_model.eval()
        test_loss, test_acc = test(
            main_model, s_dataset, device, -1, visualize, colors, "main")
        print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

    with torch.no_grad():
        main_model.eval()
        test_loss, test_acc = test(
            main_model, t_dataset, device, -1, visualize, colors, "main")
        print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

    test_accs = []
    for epoch in range(n_epochs):
        with torch.no_grad():
            pmodel.eval()
            main_model.eval()
            test_loss, test_acc = test(
                main_model, t_dataset, device, -1, visualize, colors, "prompt", pmodel)
            print(f'Epoch {epoch}/{n_epochs}, Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}', "#"*100)
            if epoch >= 125:
                test_accs.append(test_acc)

        pmodel.train()
        main_model.eval()
        total_loss = 0
        counter = 0
        x_mean = 0
        counter = 0
        for i, batch in enumerate(t_dataset.train_loader):
            optimizer.zero_grad()
            labels = batch.y
            batch = batch.to_data_list()
            # prompt_batch = [
            #     aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, aug_type = aug_type, mode = aug_mode).to(device)
            #     for g in batch
            # ]
            pos_batch = [
                Data(x=g.x, edge_index=g.edge_index, y=g.y).to(device)
                for g in batch
            ]
            prompt_batch = [
                aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, aug_type = aug_type, mode = aug_mode).to(device)
                for g in batch
            ]
            neg_batch = [
                aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, aug_type = aug_type, mode = "arbitrary").to(device)
                for g in batch
            ]
            prompt_batch = pmodel(prompt_batch)
            prompt_x_adj = batch_to_xadj_list(prompt_batch, device)
            pos_x_adj = batch_to_xadj_list(pos_batch, device)
            neg_x_adj = batch_to_xadj_list(neg_batch, device)
            prompt_out, prompt_embds = main_model(prompt_x_adj)
            pos_out, pos_embds = main_model(pos_x_adj)
            pos_probs = F.softmax(pos_out, dim=1)
            # pos_embds = torch.cat((pos_embds, torch.ones(pos_embds.size(0), 1)), dim=1)
            pos_norm = pos_embds.norm(p=2, dim=1)
            pos_norm = torch.max(pos_norm, torch.ones_like(pos_norm)*1e-8)
            pos_embds = (pos_embds.T / pos_norm).T
            initc = pos_probs.T @ pos_embds
            initc = initc / (1e-8 + pos_probs.sum(axis=0)[:,None])
            initc_norm = initc.norm(p=2, dim=1)
            initc_norm = torch.max(initc_norm, torch.ones_like(initc_norm)*1e-8)
            dists = pos_embds @ initc.T / \
             (pos_norm[:, None].tile(1, initc.size(0)) * initc_norm[None, :].tile(pos_embds.size(0), 1))
            # dd = cdist(ps_embds, initc, 'cosine')
            pred_label = dists.argmax(dim=1)

            for round in range(2):
                pos_probs = torch.eye(pos_probs.shape[1])[pred_label, :]
                initc = pos_probs.T @ pos_embds
                initc = initc / (1e-8 + pos_probs.sum(dim=0)[:,None])
                initc = torch.where(initc > 0, initc, 1e-8)
                initc_norm = initc.norm(p=2, dim=1)
                initc_norm = torch.max(initc_norm, torch.ones_like(initc_norm)*1e-8)
                dists = pos_embds @ initc.T / \
                (pos_norm[:, None].tile(1, initc.size(0)) * initc_norm[None, :].tile(pos_embds.size(0), 1))
                # dd = cdist(ps_embds, initc, 'cosine')
                pred_label = dists.argmax(dim=1)
            # ps_probs = torch.as_tensor(np.eye(K)[pred_label, :], device = device)
            pos_probs = F.softmax(dists, dim=1).detach().to(device)
            loss = F.cross_entropy(prompt_out, pos_probs)
            # softmax_out = F.softmax(prompt_out, dim=1)
            # loss += entropy_loss(softmax_out).mean()
            # b_softmax = softmax_out.mean(dim=0)
            # loss += torch.sum(b_softmax * torch.log(b_softmax + 1e-5))

            loss.backward()
            optimizer.step()
    test_average_acc = np.array(test_accs).mean()
    print("The average accuracu on test data is: ", test_average_acc)
    average_acc.append(test_average_acc)
print(f"Total after {len(average_acc)} runs: ", np.array(average_acc).mean())