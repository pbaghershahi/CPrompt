average_acc = []
for _ in range(5):

    h_dim = 64
    ph_dim = 64
    o_dim = 64
    n_layers = 2
    enc_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
    main_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    token_num = 30
    pmodel = HeavyPrompt(
        s_dataset.x.size(1),
        token_num,
        trans_x=False
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

    n_epochs = 500
    temperature = 1
    n_drops = 0.15
    batch_size = 32
    n_augs = 2
    visualize = False
    if visualize:
        colors = np.array([
            "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
            for i in range(s_dataset.y.unique().size(0))])
    else:
        colors = None

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
            labels = batch.y
            labels = labels.to(device)
            batch = batch.to_data_list()
            prompt_batch = pmodel(batch)
            prompt_x_adj = batch_to_xadj_list(prompt_batch, device)
            prompt_out, _ = main_model(prompt_x_adj)
            loss = F.cross_entropy(prompt_out, labels, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_average_acc = np.array(test_accs).mean()
    print("The average accuracu on test data is: ", test_average_acc)
    average_acc.append(test_average_acc)
print(f"Total after {len(average_acc)} runs: ", np.array(average_acc).mean())