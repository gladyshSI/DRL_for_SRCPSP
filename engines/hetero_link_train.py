import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, HeteroData

from gnn_models.hetero_gnn_link_predictor import HeteroGNN


# -------------------------
# Helpers (mostly unchanged)
# -------------------------
def build_dst_pool_and_offsets(out: dict[str, torch.Tensor], dst_types: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
    emb_list = []
    offsets = {}  # for each dst type -> first id in the pool emb_all
    offset = 0
    # if a dst type is missing in out, we append empty tensor
    for t in dst_types:
        emb = out.get(t)
        if emb is None:
            emb = torch.zeros((0, next(iter(out.values())).size(1)), device=next(iter(out.values())).device)
        emb_list.append(emb)
        offsets[t] = offset
        offset += emb.size(0)
    if len(emb_list) == 0:
        emb_all = torch.zeros((0, next(iter(out.values())).size(1)), device=next(iter(out.values())).device)
    else:
        emb_all = torch.cat(emb_list, dim=0)
    return emb_all, offsets


def make_allowed_mask(n_src: int, dst_sizes: dict[str, int], allowed_dst_types: list[str]) -> torch.Tensor:
    n_dst = sum(dst_sizes.values())
    mask = torch.zeros((n_src, n_dst), dtype=torch.bool)
    offset = 0
    for t, size in dst_sizes.items():
        if t in allowed_dst_types:
            mask[:, offset:offset+size] = True
        offset += size
    return mask


def compute_per_source_labels(batch: HeteroData, src_type: str, predict_dst_types: list[str],
                              dst_offsets: dict[str, int], n_dst: int) -> torch.LongTensor:
    device = batch[src_type].x.device
    n_src = batch[src_type].x.size(0)
    labels = torch.full((n_src,), n_dst, dtype=torch.long, device=device)  # default NULL
    for dst_t in predict_dst_types:
        key = (src_type, 'predict', dst_t)
        if key not in batch.edge_types:
            continue
        e_idx = batch[key].edge_label_index
        e_lbl = batch[key].edge_label
        if e_idx is None:
            continue
        src_idx = e_idx[0]
        dst_idx = e_idx[1]
        pos_mask = (e_lbl == 1)
        if pos_mask.sum() == 0:
            continue
        pos_src = src_idx[pos_mask]
        pos_dst = dst_idx[pos_mask]
        target_indices = pos_dst + dst_offsets[dst_t]
        labels[pos_src] = target_indices.to(device)
    return labels


# -------------------------
# Scoring helper using model.scorer (MLP)
# -------------------------
def compute_pairwise_scores_with_mlp(model: HeteroGNN, emb_src: torch.Tensor, emb_dst_all_with_null: torch.Tensor) -> torch.Tensor:
    """
    emb_src: (n_src, d)
    emb_dst_all_with_null: (n_dst_p1, d)
    returns scores: (n_src, n_dst_p1)
    """
    n_src, d = emb_src.size()
    n_dst_p1 = emb_dst_all_with_null.size(0)
    # broadcast and concat: (n_src, n_dst_p1, 2D)
    src_exp = emb_src.unsqueeze(1).expand(-1, n_dst_p1, -1)
    dst_exp = emb_dst_all_with_null.unsqueeze(0).expand(n_src, -1, -1)
    pairs = torch.cat([src_exp, dst_exp], dim=-1)  # (n_src, n_dst_p1, 2D)
    # feed through scorer: flatten -> apply -> reshape
    flat = pairs.view(-1, 2 * d)  # (n_src * n_dst_p1, 2d)
    with torch.enable_grad():
        out = model.scorer(flat)  # (n_src * n_dst_p1, 1)
    scores = out.view(n_src, n_dst_p1)
    return scores


# -------------------------
# Evaluation & training loops (modified to use MLP scorer)
# -------------------------
def evaluate(model: HeteroGNN, loader: DataLoader, device, src_configs: dict[str, list[str]]):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_dict = {t: (batch[t].x if 'x' in batch[t] else None) for t in batch.node_types}
            edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types if 'edge_index' in batch[edge_type]}
            out = model(x_dict, edge_index_dict)

            for src_type, allowed_dsts in src_configs.items():
                n_src = batch[src_type].x.size(0)
                if n_src == 0:
                    continue

                emb_dst_all, dst_offsets = build_dst_pool_and_offsets(out, allowed_dsts)
                dst_sizes = {t: (out[t].size(0) if t in out else 0) for t in allowed_dsts}
                n_dst = emb_dst_all.size(0)

                if n_dst == 0:
                    labels = compute_per_source_labels(batch, src_type, allowed_dsts, dst_offsets, n_dst)
                    total_correct += (labels == n_dst).sum().item()
                    total_count += labels.numel()
                    continue

                emb_dst_all_with_null = torch.cat([emb_dst_all, model.null_emb.expand(1, -1)], dim=0)
                emb_src = out[src_type]

                # compute pairwise scores via MLP
                scores = compute_pairwise_scores_with_mlp(model, emb_src, emb_dst_all_with_null)  # (n_src, n_dst+1)

                # mask allowed
                allowed_mask = make_allowed_mask(n_src, dst_sizes, allowed_dsts).to(scores.device)
                allowed_with_null = torch.cat([allowed_mask, torch.ones((n_src,1), dtype=torch.bool, device=scores.device)], dim=1)
                scores_masked = scores.clone()
                scores_masked[~allowed_with_null] = -1e9

                preds = scores_masked.argmax(dim=1)  # (n_src,)
                labels = compute_per_source_labels(batch, src_type, allowed_dsts, dst_offsets, n_dst)
                total_correct += (preds == labels).sum().item()
                total_count += n_src

    return total_correct / max(total_count, 1)


def train_loop(model: HeteroGNN, train_loader: DataLoader, val_loader: DataLoader,
               device, epochs=30, lr=1e-3, src_configs=None, save_path='best_model.pt'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = -1.0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        it = 0
        for batch in train_loader:
            batch = batch.to(device)
            x_dict = {t: (batch[t].x if 'x' in batch[t] else None) for t in batch.node_types}
            edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in batch.edge_types if 'edge_index' in batch[edge_type]}
            out = model(x_dict, edge_index_dict)

            loss = 0.0
            for src_type, allowed_dsts in src_configs.items():
                n_src = batch[src_type].x.size(0)
                if n_src == 0:
                    continue

                emb_dst_all, dst_offsets = build_dst_pool_and_offsets(out, allowed_dsts)
                dst_sizes = {t: (out[t].size(0) if t in out else 0) for t in allowed_dsts}
                nDst = emb_dst_all.size(0)
                labels = compute_per_source_labels(batch, src_type, allowed_dsts, dst_offsets, nDst)

                if nDst == 0:
                    continue

                emb_dst_all_with_null = torch.cat([emb_dst_all, model.null_emb.expand(1, -1)], dim=0)
                emb_src = out[src_type]

                # compute pairwise scores using the scorer MLP
                scores = compute_pairwise_scores_with_mlp(model, emb_src, emb_dst_all_with_null)  # (n_src, n_dst+1)

                allowed_mask = make_allowed_mask(n_src, dst_sizes, allowed_dsts).to(scores.device)
                allowed_with_null = torch.cat([allowed_mask, torch.ones((n_src, 1), dtype=torch.bool, device=scores.device)], dim=1)
                scores_masked = scores.clone()
                scores_masked[~allowed_with_null] = -1e9

                logp = F.log_softmax(scores_masked, dim=1)
                loss_src = F.nll_loss(logp, labels, reduction='mean')
                loss = loss + loss_src

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            it += 1

        avg_loss = total_loss / max(it, 1)
        val_acc = evaluate(model, val_loader, device, src_configs)
        print(f"Epoch {epoch:03d} | Train loss: {avg_loss:.4f} | Val acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    print("Training finished. Best val acc:", best_val)

