def cal_entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-8)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def ntxent_loss(
        prompt_score,
        pos_score,
        neg_score,
        labels=None,
        weighting=None):
    # prompt_score = F.softmax(prompt_score, dim=1)
    # pos_score = F.softmax(pos_score, dim=1)
    # neg_score = F.softmax(neg_score, dim=1)
    # pos_entropy = cal_entropy(F.softmax(pos_score, dim=1)).detach().clone()
    neg_entropy = cal_entropy(F.softmax(neg_score, dim=1)).detach().clone()
    # pos_entropy = torch.max(pos_entropy, torch.ones_like(pos_entropy)*1e-8)
    neg_entropy = torch.max(neg_entropy, torch.ones_like(neg_entropy)*1e-8)
    # pos_score = pos_score * 1/pos_entropy[:, None].tile(1, pos_score.size(1))
    neg_score = neg_score * 1/neg_entropy[:, None].tile(1, neg_score.size(1))
    pr_norm = prompt_score.norm(p=2, dim=1)
    pr_norm = torch.max(pr_norm, torch.ones_like(pr_norm)*1e-8)
    p_norm = pos_score.norm(p=2, dim=1)
    p_norm = torch.max(p_norm, torch.ones_like(p_norm)*1e-8)
    pos = (prompt_score * pos_score).sum(dim=1)/(pr_norm * p_norm)
    pr_norm = pr_norm[:, None].tile(1, pr_norm.size(0))
    n_norm = neg_score.norm(p=2, dim=1)
    n_norm = torch.max(n_norm, torch.ones_like(n_norm)*1e-8)
    n_norm = n_norm[None, :].tile(pr_norm.size(0), 1)
    neg = (prompt_score @ neg_score.T) / (pr_norm * n_norm)

    if labels is not None:
        # print("labels are being used!")
        min_neg = labels.size(0) + 1
        neg_labels = torch.zeros((labels.size(0), labels.size(0))).bool()
        for i in range(labels.size(0)):
            neg_labels[i, :] = (labels != labels[i])
        min_neg = neg_labels.sum(dim=0).min()
        idx_mask = torch.arange(neg.size(1)).tile(neg.size(0), 1)
        idx_mask = torch.where(neg_labels, idx_mask, neg.size(1)+1)
        idx_mask = idx_mask.sort(dim=1, descending=False)[0][:, :min_neg]
        neg = neg.gather(1, idx_mask)

    neg = torch.cat([neg, pos[:, None]], dim=1)
    neg = torch.logsumexp(neg, dim=1)
    # print(neg)
    # print(neg.isnan().any())
    # print((-pos + neg).isnan().any())
    loss = (-pos + neg).mean()
    # print(loss)
    return loss