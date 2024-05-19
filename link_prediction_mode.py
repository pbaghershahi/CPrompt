def graph_collate(batch):
    if not isinstance(batch, list):
        x_list = []
        for g in batch:
            x_list.extend(g)
    else:
        x_list = batch
    x_batch = torch.cat(x_list, dim=0)
    return x_batch
    

class GDataset(nn.Module):
    def __init__(self,):
        pass

    def init_loaders_(self,):
        "Implement this is children classes"
        pass

    def normalize_feats_(self,):
        "Implement this is children classes"
        return "x"

    def init_ds_idxs_(self, train_idxs, valid_idxs, test_idxs, train_test_split, shuffle, seed):
        if (train_idxs is not None) and (valid_idxs is not None) and (test_idxs is not None):
            self.n_train = train_idxs.size(0)
            self.n_valid = valid_idxs.size(0)
            self.n_test = test_idxs.size(0)
            self.train_idxs = train_idxs
            self.valid_idxs = valid_idxs
            self.test_idxs = test_idxs
        else:
            all_idxs = torch.arange(self.num_gsamples)
            if shuffle: 
                fix_seed(seed)
                perm = torch.randperm(self.num_gsamples)
                all_idxs = all_idxs[perm]
            if train_test_split[0] + train_test_split[1] != 1.0:
                valid_per = 1 - (train_test_split[0] + train_test_split[1])
            else:
                valid_per = 0.0
            self.n_train = int(self.num_gsamples * train_test_split[0])
            self.n_valid = int(self.num_gsamples * valid_per)
            self.n_test = self.num_gsamples - (self.n_train + self.n_valid)
            self.train_idxs = all_idxs[:self.n_train]
            self.valid_idxs = all_idxs[self.n_train:self.n_train + self.n_valid]
            self.test_idxs = all_idxs[self.n_train + self.n_valid:self.n_train + self.n_valid + self.n_test]

    def initialize(self,):
        "Implement this is children classes"
        return "x"


class SimpleDataset(Dataset):
    def __init__(self,
                 feats: list,
                 **kwargs) -> None:
        super(SimpleDataset, self).__init__()
        self.x = feats

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]


class NodeToGraphDataset(GDataset):
    def __init__(self,
                 main_dataset,
                 n_hopes = 2,
                 **kwargs) -> None:
        super(NodeToGraphDataset, self).__init__()
        if isinstance(main_dataset, PyG_Dataset):
            self._data = main_dataset._data
        elif isinstance(main_dataset, Data):
            self._data = main_dataset
        else:
            raise "Data type is not supported!"
        self.n_feats = main_dataset.x.size(1)
        self.num_nsamples = main_dataset.x.size(0)
        self.num_nclass = main_dataset.y.unique().size(0)
        self.num_gclass = self.num_nclass
        self.num_gsamples = self.num_nsamples
        self.n_hopes = n_hopes

    @property
    def x(self,):
        return self._data.x

    def normalize_feats_(self, normalize_mode, **kwargs):
        self.train_ds.x, train_normal_params = normalize_(self.train_ds.x, dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self.valid_ds.x, _ = normalize_(
                self.valid_ds.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds.x, _ = normalize_(
                self.test_ds.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)

    def init_loaders_(self, loader_collate, batch_size):
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)

    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split = [0.85, 0.15],
            loader_collate = graph_collate,
            batch_size = 32, 
            normalize_mode = None,
            shuffle = False, **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        self.train_ds = SimpleDataset(self._data.x[self.train_idxs])
        self.valid_ds = SimpleDataset(self._data.x[self.valid_idxs])
        self.test_ds = SimpleDataset(self._data.x[self.test_idxs])

        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.init_loaders_(loader_collate, batch_size)


class PretrainedModel(nn.Module):
    def __init__(self, d_feat, d_hid, d_class, n_layers, r_dropout, *args, **kwargs) -> None:
        super(PretrainedModel, self).__init__(*args, **kwargs)
        self.gnn_modul = GCN(
            in_channels = d_feat,
            hidden_channels = d_hid,
            num_layers = n_layers,
            out_channels = d_hid,
            dropout = .2,
            act = "relu",
            norm = None
        )
        self.r_dropout = r_dropout
        self.decoder = nn.Linear(d_hid, d_class)

    def forward(self, x, edge_index, decoder = True, device = None):
        if not decoder:
            scores = self.decoder(x)
            return scores, "embeds"
        x = self.gnn_modul(x, edge_index)
        scores = self.decoder(F.dropout(x, p=self.r_dropout, training=self.training))
        return scores, x