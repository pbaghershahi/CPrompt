average_acc = []
for _ in range(5):

    h_dim = 64
    ph_dim = 64
    o_dim = 64
    n_layers = 2
    n_epochs = 150
    temperature = 1
    p_raug = 0.15
    n_raug = 0.15
    batch_size = 32
    n_augs = 2
    num_neg = 5
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
                aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), p_raug, aug_type = aug_type, mode = aug_mode).to(device)
                for g in batch
            ]
            # ipdb.set_trace()
            neg_batch = [
                aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_raug, aug_type = aug_type, mode = "arbitrary").to(device)
                for g in batch for _ in range(num_neg)
            ]
            prompt_batch = pmodel(prompt_batch)
            prompt_x_adj = batch_to_xadj_list(prompt_batch, device)
            pos_x_adj = batch_to_xadj_list(pos_batch, device)
            neg_x_adj = batch_to_xadj_list(neg_batch, device)
            prompt_out, prompt_embed = main_model(prompt_x_adj)
            pos_out, pos_embed = main_model(pos_x_adj)
            neg_out, neg_embed = main_model(neg_x_adj)
            neg_jsds = torch.zeros((len(batch), num_neg), dtype=torch.float, device=neg_out.device)
            neg_out_reshaped = neg_out.view(len(batch), num_neg, -1).detach().clone()
            for k in range(prompt_out.size(0)):
                for l in range(num_neg):
                    neg_jsds[k, l] = cal_JSD(prompt_out[k, :], neg_out_reshaped[k, l, :])
            _, argmax_loss = neg_jsds.sort(dim=1, descending=False)
            argmax_loss = ((argmax_loss + (torch.arange(len(batch)).to(device) * num_neg)[:, None])[:, :1]).reshape(-1)
            neg_out = neg_out[argmax_loss, :].view(len(batch), 1, -1).mean(dim=1)
            # loss = multiclass_triplet_loss(prompt_out, pos_out, neg_out, 1)
            loss = ntxent_loss(prompt_out, pos_out, neg_out, weighting=None)
            if add_link_loss:
                adj_labels = get_adj_labels(batch).to(device)
                loss += link_predict_loss(prompt_batch, adj_labels)
            loss.backward()
            optimizer.step()

    test_average_acc = np.array(test_accs).mean()
    print("The average accuracu on test data is: ", test_average_acc)
    average_acc.append(test_average_acc)
print(f"Total after {len(average_acc)} runs: ", np.array(average_acc).mean())