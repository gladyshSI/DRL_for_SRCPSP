import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Utilities
# ============================================================

def _zero(device):
    return torch.tensor(0.0, device=device)


def _safe_concat(values):
    if len(values) == 0:
        return np.array([])
    return np.concatenate(values)


def _to_numpy(x):
    return x.detach().cpu().view(-1).numpy()


def _safe_mean(values):
    values = [v for v in values if not math.isnan(v)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _has_two_classes(y_true):
    return len(np.unique(y_true)) == 2


def classification_metrics(y_true, y_pred, y_prob=None):
    if len(y_true) == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "balanced_acc": float("nan"),
            "mcc": float("nan"),
            "roc_auc": float("nan"),
            "ap": float("nan"),
            "brier": float("nan"),
            "pos_rate": float("nan"),
            "pred_rate": float("nan"),
        }

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    if _has_two_classes(y_true):
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        if y_prob is not None:
            roc_auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        else:
            roc_auc = float("nan")
            ap = float("nan")
    else:
        balanced_acc = float("nan")
        mcc = float("nan")
        roc_auc = float("nan")
        ap = float("nan")

    if y_prob is not None:
        brier = brier_score_loss(y_true, y_prob)
    else:
        brier = float("nan")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "ap": ap,
        "brier": brier,
        "pos_rate": float(np.mean(y_true)),
        "pred_rate": float(np.mean(y_pred)),
    }


def regression_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "medae": float("nan"),
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    medae = float(np.median(np.abs(y_true - y_pred)))

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "medae": medae,
    }


def format_float(x, digits=4):
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


# ============================================================
# Positive-class weights
# ============================================================

def compute_pos_weights(loader, src_configs, device):
    pos_counts = defaultdict(float)
    total_counts = defaultdict(float)

    for batch in loader:
        for src_type, dst_list in src_configs.items():
            for dst_type in dst_list:
                edge_type = (src_type, "predict", dst_type)

                if edge_type not in batch.edge_types:
                    continue

                if "edge_value" not in batch[edge_type]:
                    continue

                labels = batch[edge_type].edge_value
                pos_counts[edge_type] += labels.sum().item()
                total_counts[edge_type] += labels.numel()

    pos_weights = {}

    for edge_type in total_counts:
        pos = pos_counts[edge_type]
        total = total_counts[edge_type]
        neg = total - pos

        if pos > 0:
            # sqrt is less aggressive than neg / pos
            weight = math.sqrt(neg / pos)
        else:
            weight = 1.0

        pos_weights[edge_type] = torch.tensor(weight, device=device)

    return pos_weights


# ============================================================
# Edge scoring
# ============================================================

def compute_edge_scores(model, out, batch, edge_type):
    src_type, _, dst_type = edge_type

    edge_label_index = batch[edge_type].edge_label_index

    labels = batch[edge_type].edge_value.float()
    buffers = batch[edge_type].edge_length.float()

    src = out[src_type][edge_label_index[0]]
    dst = out[dst_type][edge_label_index[1]]

    pair = torch.cat([src, dst], dim=-1)

    logits = model.link_scorer(pair).squeeze(-1)

    # Non-negative length prediction
    pred_buf = F.softplus(model.buf_scorer(pair).squeeze(-1))

    return logits, labels, pred_buf, buffers


def compute_edge_scores_dif_heads(model, out, batch, edge_type):
    src_type, _, dst_type = edge_type

    edge_label_index = batch[edge_type].edge_label_index

    labels = batch[edge_type].edge_value.float()
    buffers = batch[edge_type].edge_length.float()

    src = out[src_type][edge_label_index[0]]
    dst = out[dst_type][edge_label_index[1]]

    pair = torch.cat([src, dst], dim=-1)

    key = model.edge_type_to_str(edge_type)

    if key not in model.link_scorers:
        raise KeyError(
            f"No relation-specific scorer for edge_type={edge_type}. "
            f"Available: {list(model.link_scorers.keys())}"
        )

    logits = model.link_scorers[key](pair).squeeze(-1)

    pred_buf = F.softplus(
        model.buf_scorers[key](pair).squeeze(-1)
    )

    return logits, labels, pred_buf, buffers


def collect_edge_outputs(model, out, batch, src_configs):
    """
    Computes all candidate edge logits first.

    This is important because assignment-aware losses need to see
    competing edges across relation types.
    """
    outputs = {}

    for src_type, dst_list in src_configs.items():
        for dst_type in dst_list:
            edge_type = (src_type, "predict", dst_type)

            if edge_type not in batch.edge_types:
                continue

            if "edge_label_index" not in batch[edge_type]:
                continue

            #TODO: Change for one or dif heads
            logits, labels, pred_buf, buf = compute_edge_scores_dif_heads(  # compute_edge_scores(
                model,
                out,
                batch,
                edge_type,
            )

            if logits.numel() == 0:
                continue

            outputs[edge_type] = {
                "logits": logits.view(-1),
                "labels": labels.view(-1).float(),
                "pred_buf": pred_buf.view(-1),
                "buf": buf.view(-1).float(),
                "edge_label_index": batch[edge_type].edge_label_index,
            }

    return outputs


# ============================================================
# Greedy matching
# ============================================================

def greedy_matching_preds_across_edge_types(
    edge_label_index_by_type,
    probs_by_type,
    min_score=0.5,
):
    """
    Greedy bipartite matching across multiple edge types.

    Guarantees:
    - each source node is used at most once
    - each destination node is used at most once
    - destination uniqueness is shared across edge types
    """
    preds_by_type = {
        edge_type: torch.zeros_like(probs, dtype=torch.long)
        for edge_type, probs in probs_by_type.items()
    }

    candidates = []

    for edge_type, probs in probs_by_type.items():
        src_type, _, dst_type = edge_type

        edge_label_index = edge_label_index_by_type[edge_type]
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]

        probs = probs.view(-1)

        for i in range(probs.numel()):
            score = probs[i].item()

            if min_score is not None and score < min_score:
                continue

            candidates.append(
                (
                    score,
                    edge_type,
                    i,
                    src_type,
                    int(src_nodes[i].item()),
                    dst_type,
                    int(dst_nodes[i].item()),
                )
            )

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_sources = set()
    used_destinations = set()

    for _, edge_type, i, src_type, src_id, dst_type, dst_id in candidates:
        src_key = (src_type, src_id)
        dst_key = (dst_type, dst_id)

        if src_key in used_sources:
            continue

        if dst_key in used_destinations:
            continue

        preds_by_type[edge_type][i] = 1
        used_sources.add(src_key)
        used_destinations.add(dst_key)

    return preds_by_type


# ============================================================
# Assignment-aware training losses
# ============================================================

def bce_classification_loss(outputs, pos_weights):
    losses = []

    for edge_type, values in outputs.items():
        logits = values["logits"]
        labels = values["labels"]

        pos_weight = pos_weights.get(
            edge_type,
            torch.tensor(1.0, device=logits.device),
        )

        losses.append(
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
            )
        )

    if len(losses) == 0:
        return None

    return torch.stack(losses).mean()


def positive_only_regression_loss(outputs):
    """
    Length is meaningful only for true links.

    This avoids training the length head to predict zero/small values
    on negative candidate edges.
    """
    losses = []

    for _, values in outputs.items():
        labels = values["labels"]
        pred_buf = values["pred_buf"]
        buf = values["buf"]

        pos_mask = labels == 1

        if pos_mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_buf[pos_mask],
                    buf[pos_mask],
                )
            )

    if len(losses) == 0:
        device = next(iter(outputs.values()))["logits"].device
        return _zero(device)

    return torch.stack(losses).mean()


def soft_assignment_conflict_loss(batch, outputs, temperature=1.0):
    """
    Penalizes soft degree > 1.

    This makes the raw model less dependent on greedy cleanup.

    For every source:
        sum p(edge from source) <= 1

    For every destination:
        sum p(edge into destination) <= 1

    Destination uniqueness is shared across edge types.
    """
    if len(outputs) == 0:
        return None

    device = next(iter(outputs.values()))["logits"].device

    losses = []
    dst_mass_by_type = {}

    for edge_type, values in outputs.items():
        src_type, _, dst_type = edge_type

        logits = values["logits"]
        edge_label_index = values["edge_label_index"]

        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        probs = torch.sigmoid(logits / temperature)

        # Source degree mass
        src_mass = probs.new_zeros(batch[src_type].num_nodes)
        src_mass.scatter_add_(0, src_idx, probs)

        src_violation = F.relu(src_mass - 1.0).pow(2)
        losses.append(src_violation.mean())

        # Destination degree mass, shared across left/right edge types
        if dst_type not in dst_mass_by_type:
            dst_mass_by_type[dst_type] = probs.new_zeros(batch[dst_type].num_nodes)

        dst_mass_by_type[dst_type].scatter_add_(0, dst_idx, probs)

    for _, dst_mass in dst_mass_by_type.items():
        dst_violation = F.relu(dst_mass - 1.0).pow(2)
        losses.append(dst_violation.mean())

    if len(losses) == 0:
        return _zero(device)

    return torch.stack(losses).mean()


def soft_count_loss(outputs):
    """
    Encourages the model to predict roughly the right number of links.

    Without this, the conflict loss can be satisfied by lowering every score.
    """
    if len(outputs) == 0:
        return None

    device = next(iter(outputs.values()))["logits"].device

    pred_count = _zero(device)
    true_count = _zero(device)

    for _, values in outputs.items():
        probs = torch.sigmoid(values["logits"])
        labels = values["labels"]

        pred_count = pred_count + probs.sum()
        true_count = true_count + labels.sum()

    denom = true_count.detach().clamp(min=1.0)

    return F.smooth_l1_loss(
        pred_count / denom,
        true_count / denom,
    )


def conflict_ranking_loss(batch, outputs, margin=1.0, topk=10):
    """
    For every true edge, score it above hard negative edges that would
    compete with it in greedy matching.

    Conflicts are:
    - same source node
    - same destination node
    """
    if len(outputs) == 0:
        return None

    device = next(iter(outputs.values()))["logits"].device

    src_offsets = {}
    dst_offsets = {}

    next_src_offset = 0
    next_dst_offset = 0

    for edge_type in outputs:
        src_type, _, dst_type = edge_type

        if src_type not in src_offsets:
            src_offsets[src_type] = next_src_offset
            next_src_offset += batch[src_type].num_nodes

        if dst_type not in dst_offsets:
            dst_offsets[dst_type] = next_dst_offset
            next_dst_offset += batch[dst_type].num_nodes

    all_scores = []
    all_labels = []
    all_src_groups = []
    all_dst_groups = []

    for edge_type, values in outputs.items():
        src_type, _, dst_type = edge_type
        edge_label_index = values["edge_label_index"]

        scores = values["logits"].view(-1)
        labels = values["labels"].view(-1).long()

        src_groups = edge_label_index[0] + src_offsets[src_type]
        dst_groups = edge_label_index[1] + dst_offsets[dst_type]

        all_scores.append(scores)
        all_labels.append(labels)
        all_src_groups.append(src_groups)
        all_dst_groups.append(dst_groups)

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    src_groups = torch.cat(all_src_groups)
    dst_groups = torch.cat(all_dst_groups)

    pos_indices = torch.where(labels == 1)[0]

    losses = []

    for pos_i in pos_indices:
        same_src = src_groups == src_groups[pos_i]
        same_dst = dst_groups == dst_groups[pos_i]

        conflict_neg_mask = (same_src | same_dst) & (labels == 0)
        neg_scores = scores[conflict_neg_mask]

        if neg_scores.numel() == 0:
            continue

        k = min(topk, neg_scores.numel())
        hard_neg_scores = torch.topk(neg_scores, k=k).values

        pos_score = scores[pos_i]

        losses.append(
            F.relu(margin - pos_score + hard_neg_scores).mean()
        )

    if len(losses) == 0:
        return _zero(device)

    return torch.stack(losses).mean()


def greedy_error_loss(
    batch,
    outputs,
    threshold=0.3,
    false_positive_weight=2.0,
    false_negative_weight=2.0,
):
    """
    Runs greedy matching with detached probabilities, then applies
    extra BCE loss to the mistakes made by greedy.

    This is not fully differentiable through greedy matching, but it is
    useful hard-example mining.
    """
    if len(outputs) == 0:
        return None

    device = next(iter(outputs.values()))["logits"].device

    probs_by_type = {
        edge_type: torch.sigmoid(values["logits"].detach())
        for edge_type, values in outputs.items()
    }

    edge_label_index_by_type = {
        edge_type: values["edge_label_index"]
        for edge_type, values in outputs.items()
    }

    preds_by_type = greedy_matching_preds_across_edge_types(
        edge_label_index_by_type=edge_label_index_by_type,
        probs_by_type=probs_by_type,
        min_score=threshold,
    )

    losses = []

    for edge_type, values in outputs.items():
        logits = values["logits"]
        labels = values["labels"].float().view(-1)
        preds = preds_by_type[edge_type].float().view(-1)

        false_selected = (preds == 1) & (labels == 0)
        missed_true = (preds == 0) & (labels == 1)

        if false_selected.any():
            losses.append(
                false_positive_weight
                * F.binary_cross_entropy_with_logits(
                    logits[false_selected],
                    torch.zeros_like(logits[false_selected]),
                )
            )

        if missed_true.any():
            losses.append(
                false_negative_weight
                * F.binary_cross_entropy_with_logits(
                    logits[missed_true],
                    torch.ones_like(logits[missed_true]),
                )
            )

    if len(losses) == 0:
        return _zero(device)

    return torch.stack(losses).mean()


# ============================================================
# Diagnostics
# ============================================================

def degree_violation_stats(batch, outputs, preds_by_type):
    """
    Counts degree > 1 violations for a set of binary predictions.

    Works for raw thresholded predictions and for matched predictions.
    """
    source_violations = 0
    destination_violations = 0

    total_source_nodes = 0
    total_destination_nodes = 0

    max_source_degree = 0
    max_destination_degree = 0

    dst_degree_by_type = {}

    for edge_type, values in outputs.items():
        src_type, _, dst_type = edge_type

        edge_label_index = values["edge_label_index"]
        preds = preds_by_type[edge_type].view(-1).float()

        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        # Source degrees are unique per source type
        src_degree = preds.new_zeros(batch[src_type].num_nodes)
        src_degree.scatter_add_(0, src_idx, preds)

        source_violations += int((src_degree > 1).sum().item())
        total_source_nodes += int(src_degree.numel())
        max_source_degree = max(max_source_degree, int(src_degree.max().item()) if src_degree.numel() else 0)

        # Destination degrees are shared across edge types with same destination type
        if dst_type not in dst_degree_by_type:
            dst_degree_by_type[dst_type] = preds.new_zeros(batch[dst_type].num_nodes)

        dst_degree_by_type[dst_type].scatter_add_(0, dst_idx, preds)

    for _, dst_degree in dst_degree_by_type.items():
        destination_violations += int((dst_degree > 1).sum().item())
        total_destination_nodes += int(dst_degree.numel())
        max_destination_degree = max(
            max_destination_degree,
            int(dst_degree.max().item()) if dst_degree.numel() else 0,
        )

    return {
        "source_violations": source_violations,
        "destination_violations": destination_violations,
        "total_source_nodes": total_source_nodes,
        "total_destination_nodes": total_destination_nodes,
        "max_source_degree": max_source_degree,
        "max_destination_degree": max_destination_degree,
    }


def graph_exact_match_stats(batch, outputs, preds_by_type):
    """
    Computes exact assignment match per graph if PyG batch vectors exist.

    Returns:
        correct_graphs, total_graphs

    If batch vectors are unavailable, returns 0, 0.
    """
    graph_ids = []

    for node_type in batch.node_types:
        if hasattr(batch[node_type], "batch"):
            graph_ids.append(batch[node_type].batch)

    if len(graph_ids) == 0:
        return 0, 0

    num_graphs = max(int(g.max().item()) for g in graph_ids if g.numel() > 0) + 1

    true_sets = [set() for _ in range(num_graphs)]
    pred_sets = [set() for _ in range(num_graphs)]

    usable = False

    for edge_type, values in outputs.items():
        src_type, _, dst_type = edge_type

        if not hasattr(batch[src_type], "batch"):
            continue

        usable = True

        edge_label_index = values["edge_label_index"]
        labels = values["labels"].view(-1).long()
        preds = preds_by_type[edge_type].view(-1).long()

        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]

        src_graph = batch[src_type].batch[src_idx]

        for i in range(labels.numel()):
            g = int(src_graph[i].item())

            edge_key = (
                str(edge_type),
                int(src_idx[i].item()),
                int(dst_idx[i].item()),
            )

            if labels[i].item() == 1:
                true_sets[g].add(edge_key)

            if preds[i].item() == 1:
                pred_sets[g].add(edge_key)

    if not usable:
        return 0, 0

    correct = sum(1 for g in range(num_graphs) if true_sets[g] == pred_sets[g])

    return correct, num_graphs


# ============================================================
# Training
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    device,
    src_configs,
    epochs=20,
    threshold=0.3,
    reg_loss_weight=0.01,
    lambda_conflict=0.03,
    lambda_rank=0.20,
    lambda_greedy=0.1,
    lambda_count=0.01,
    rank_margin=1.0,
    rank_topk=10,
    conflict_temperature=0.7,
    save_path="../checkpoints/best_hetero_assignment_mlp.pt",
):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_weights = compute_pos_weights(train_loader, src_configs, device)

    best_val_f1 = -float("inf")
    best_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0.0
        num_batches = 0

        loss_sums = defaultdict(float)

        relation_stats = defaultdict(lambda: {
            "count": 0,
            "pos_rate": 0.0,
            "raw_pred_rate": 0.0,
            "prob_pos": 0.0,
            "prob_neg": 0.0,
            "loss_cls": 0.0,
        })

        raw_degree_source_violations = 0
        raw_degree_destination_violations = 0
        raw_max_source_degree = 0
        raw_max_destination_degree = 0

        for batch in train_loader:
            batch = batch.to(device)

            out = model(batch)
            outputs = collect_edge_outputs(model, out, batch, src_configs)

            if len(outputs) == 0:
                continue

            loss_cls = bce_classification_loss(outputs, pos_weights)
            loss_reg = positive_only_regression_loss(outputs)

            loss_conflict = soft_assignment_conflict_loss(
                batch,
                outputs,
                temperature=conflict_temperature,
            )

            loss_rank = conflict_ranking_loss(
                batch,
                outputs,
                margin=rank_margin,
                topk=rank_topk,
            )

            loss_greedy = greedy_error_loss(
                batch,
                outputs,
                threshold=threshold,
            )

            loss_count = soft_count_loss(outputs)

            loss = (
                loss_cls
                + reg_loss_weight * loss_reg
                + lambda_conflict * loss_conflict
                + lambda_rank * loss_rank
                + lambda_greedy * loss_greedy
                + lambda_count * loss_count
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            epoch_loss += loss.item()
            num_batches += 1

            loss_sums["total"] += loss.item()
            loss_sums["cls"] += loss_cls.item()
            loss_sums["reg"] += loss_reg.item()
            loss_sums["conflict"] += loss_conflict.item()
            loss_sums["rank"] += loss_rank.item()
            loss_sums["greedy"] += loss_greedy.item()
            loss_sums["count"] += loss_count.item()

            # -------------------------
            # Training diagnostics
            # -------------------------
            raw_preds_by_type = {}

            for edge_type, values in outputs.items():
                logits = values["logits"]
                labels = values["labels"]

                probs = torch.sigmoid(logits)
                raw_preds = (probs > threshold).long()

                raw_preds_by_type[edge_type] = raw_preds

                key = str(edge_type)
                relation_stats[key]["count"] += 1
                relation_stats[key]["pos_rate"] += labels.mean().item()
                relation_stats[key]["raw_pred_rate"] += raw_preds.float().mean().item()

                pos_mask = labels == 1
                neg_mask = labels == 0

                if pos_mask.any():
                    relation_stats[key]["prob_pos"] += probs[pos_mask].mean().item()

                if neg_mask.any():
                    relation_stats[key]["prob_neg"] += probs[neg_mask].mean().item()

                pos_weight = pos_weights.get(
                    edge_type,
                    torch.tensor(1.0, device=device),
                )

                rel_cls = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                )
                relation_stats[key]["loss_cls"] += rel_cls.item()

            deg_stats = degree_violation_stats(batch, outputs, raw_preds_by_type)

            raw_degree_source_violations += deg_stats["source_violations"]
            raw_degree_destination_violations += deg_stats["destination_violations"]
            raw_max_source_degree = max(raw_max_source_degree, deg_stats["max_source_degree"])
            raw_max_destination_degree = max(raw_max_destination_degree, deg_stats["max_destination_degree"])

        if num_batches == 0:
            print(f"Epoch {epoch}: no valid batches")
            continue

        print("\n" + "=" * 90)
        print(f"Epoch {epoch}")

        print(
            "Train losses:",
            f"total={loss_sums['total'] / num_batches:.4f}",
            f"cls={loss_sums['cls'] / num_batches:.4f}",
            f"reg={loss_sums['reg'] / num_batches:.4f}",
            f"conflict={loss_sums['conflict'] / num_batches:.4f}",
            f"rank={loss_sums['rank'] / num_batches:.4f}",
            f"greedy={loss_sums['greedy'] / num_batches:.4f}",
            f"count={loss_sums['count'] / num_batches:.4f}",
        )

        print(
            "Train raw degree violations:",
            f"source={raw_degree_source_violations}",
            f"destination={raw_degree_destination_violations}",
            f"max_source_degree={raw_max_source_degree}",
            f"max_destination_degree={raw_max_destination_degree}",
        )

        print("Train relation diagnostics:")
        for key, stats in relation_stats.items():
            c = max(stats["count"], 1)

            print(
                key,
                f"loss_cls={stats['loss_cls'] / c:.4f}",
                f"pos_rate={stats['pos_rate'] / c:.4f}",
                f"raw_pred_rate={stats['raw_pred_rate'] / c:.4f}",
                f"p_pos={stats['prob_pos'] / c:.4f}",
                f"p_neg={stats['prob_neg'] / c:.4f}",
            )

        val_acc, val_f1, val_r2 = evaluate(
            model,
            val_loader,
            device,
            src_configs,
            threshold=threshold,
            split_name="Validation",
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

            print(
                f"Saved best model:",
                f"epoch={best_epoch}",
                f"val_f1={best_val_f1:.4f}",
                f"val_acc={val_acc:.4f}",
                f"val_r2={format_float(val_r2)}",
            )

    print(
        "\nTraining finished.",
        f"Best epoch={best_epoch}",
        f"best_val_f1={best_val_f1:.4f}",
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    device,
    src_configs,
    threshold=0.5,
    split_name="Evaluation",
):
    model.eval()

    # Per edge type
    raw_true = defaultdict(list)
    raw_pred = defaultdict(list)
    raw_prob = defaultdict(list)

    matched_true = defaultdict(list)
    matched_pred = defaultdict(list)

    buf_true = defaultdict(list)
    buf_pred = defaultdict(list)

    # Overall
    all_raw_true = []
    all_raw_pred = []
    all_raw_prob = []

    all_matched_true = []
    all_matched_pred = []

    all_buf_true = []
    all_buf_pred = []

    total_true_edges = 0
    total_raw_selected_edges = 0
    total_matched_selected_edges = 0

    raw_source_violations = 0
    raw_destination_violations = 0
    raw_max_source_degree = 0
    raw_max_destination_degree = 0

    matched_source_violations = 0
    matched_destination_violations = 0
    matched_max_source_degree = 0
    matched_max_destination_degree = 0

    exact_correct_graphs = 0
    exact_total_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch)
            outputs = collect_edge_outputs(model, out, batch, src_configs)

            if len(outputs) == 0:
                continue

            probs_by_type = {
                edge_type: torch.sigmoid(values["logits"])
                for edge_type, values in outputs.items()
            }

            edge_label_index_by_type = {
                edge_type: values["edge_label_index"]
                for edge_type, values in outputs.items()
            }

            raw_preds_by_type = {
                edge_type: (probs > threshold).long()
                for edge_type, probs in probs_by_type.items()
            }

            matched_preds_by_type = greedy_matching_preds_across_edge_types(
                edge_label_index_by_type=edge_label_index_by_type,
                probs_by_type=probs_by_type,
                min_score=threshold,
            )

            raw_deg = degree_violation_stats(batch, outputs, raw_preds_by_type)
            matched_deg = degree_violation_stats(batch, outputs, matched_preds_by_type)

            raw_source_violations += raw_deg["source_violations"]
            raw_destination_violations += raw_deg["destination_violations"]
            raw_max_source_degree = max(raw_max_source_degree, raw_deg["max_source_degree"])
            raw_max_destination_degree = max(raw_max_destination_degree, raw_deg["max_destination_degree"])

            matched_source_violations += matched_deg["source_violations"]
            matched_destination_violations += matched_deg["destination_violations"]
            matched_max_source_degree = max(matched_max_source_degree, matched_deg["max_source_degree"])
            matched_max_destination_degree = max(matched_max_destination_degree, matched_deg["max_destination_degree"])

            graph_correct, graph_total = graph_exact_match_stats(
                batch,
                outputs,
                matched_preds_by_type,
            )

            exact_correct_graphs += graph_correct
            exact_total_graphs += graph_total

            for edge_type, values in outputs.items():
                labels = values["labels"].view(-1).long()
                probs = probs_by_type[edge_type].view(-1)
                raw_preds = raw_preds_by_type[edge_type].view(-1)
                matched_preds = matched_preds_by_type[edge_type].view(-1)

                pred_buf = values["pred_buf"].view(-1)
                buf = values["buf"].view(-1)

                y_true = _to_numpy(labels)
                y_prob = _to_numpy(probs)
                y_raw = _to_numpy(raw_preds)
                y_matched = _to_numpy(matched_preds)

                raw_true[edge_type].append(y_true)
                raw_pred[edge_type].append(y_raw)
                raw_prob[edge_type].append(y_prob)

                matched_true[edge_type].append(y_true)
                matched_pred[edge_type].append(y_matched)

                all_raw_true.append(y_true)
                all_raw_pred.append(y_raw)
                all_raw_prob.append(y_prob)

                all_matched_true.append(y_true)
                all_matched_pred.append(y_matched)

                total_true_edges += int(labels.sum().item())
                total_raw_selected_edges += int(raw_preds.sum().item())
                total_matched_selected_edges += int(matched_preds.sum().item())

                # Length metrics only on true positive edges
                pos_mask = labels == 1

                if pos_mask.any():
                    b_true = _to_numpy(buf[pos_mask])
                    b_pred = _to_numpy(pred_buf[pos_mask])

                    buf_true[edge_type].append(b_true)
                    buf_pred[edge_type].append(b_pred)

                    all_buf_true.append(b_true)
                    all_buf_pred.append(b_pred)

    print("\n" + "-" * 90)
    print(f"{split_name} metrics @ threshold={threshold}")

    print("\nPer edge type:")
    for edge_type in sorted(matched_true.keys(), key=str):
        y_true = _safe_concat(matched_true[edge_type])

        y_raw = _safe_concat(raw_pred[edge_type])
        y_prob = _safe_concat(raw_prob[edge_type])
        y_matched = _safe_concat(matched_pred[edge_type])

        raw_m = classification_metrics(y_true, y_raw, y_prob)
        matched_m = classification_metrics(y_true, y_matched, None)

        y_buf_true = _safe_concat(buf_true[edge_type])
        y_buf_pred = _safe_concat(buf_pred[edge_type])
        reg_m = regression_metrics(y_buf_true, y_buf_pred)

        print(
            edge_type,
            f"pos_rate={format_float(raw_m['pos_rate'])}",
            f"raw_pred_rate={format_float(raw_m['pred_rate'])}",
            f"matched_pred_rate={format_float(matched_m['pred_rate'])}",
            f"raw_ap={format_float(raw_m['ap'])}",
            f"raw_auc={format_float(raw_m['roc_auc'])}",
            f"matched_p={format_float(matched_m['precision'])}",
            f"matched_r={format_float(matched_m['recall'])}",
            f"matched_f1={format_float(matched_m['f1'])}",
            f"mae_buf={format_float(reg_m['mae'])}",
            f"rmse_buf={format_float(reg_m['rmse'])}",
            f"r2_buf={format_float(reg_m['r2'])}",
        )

    y_true_all = _safe_concat(all_matched_true)

    y_raw_all = _safe_concat(all_raw_pred)
    y_prob_all = _safe_concat(all_raw_prob)

    y_matched_all = _safe_concat(all_matched_pred)

    raw_overall = classification_metrics(y_true_all, y_raw_all, y_prob_all)
    matched_overall = classification_metrics(y_true_all, y_matched_all, None)

    y_buf_true_all = _safe_concat(all_buf_true)
    y_buf_pred_all = _safe_concat(all_buf_pred)

    reg_overall = regression_metrics(y_buf_true_all, y_buf_pred_all)

    if exact_total_graphs > 0:
        exact_assignment_acc = exact_correct_graphs / exact_total_graphs
    else:
        exact_assignment_acc = float("nan")

    print("\nOverall raw edge metrics before greedy:")
    print(
        f"accuracy={format_float(raw_overall['acc'])}",
        f"precision={format_float(raw_overall['precision'])}",
        f"recall={format_float(raw_overall['recall'])}",
        f"f1={format_float(raw_overall['f1'])}",
        f"balanced_acc={format_float(raw_overall['balanced_acc'])}",
        f"mcc={format_float(raw_overall['mcc'])}",
        f"ap={format_float(raw_overall['ap'])}",
        f"auc={format_float(raw_overall['roc_auc'])}",
        f"brier={format_float(raw_overall['brier'])}",
    )

    print("\nOverall matched edge metrics after greedy:")
    print(
        f"accuracy={format_float(matched_overall['acc'])}",
        f"precision={format_float(matched_overall['precision'])}",
        f"recall={format_float(matched_overall['recall'])}",
        f"f1={format_float(matched_overall['f1'])}",
        f"balanced_acc={format_float(matched_overall['balanced_acc'])}",
        f"mcc={format_float(matched_overall['mcc'])}",
    )

    print("\nAssignment counts:")
    print(
        f"true_edges={total_true_edges}",
        f"raw_selected_edges={total_raw_selected_edges}",
        f"matched_selected_edges={total_matched_selected_edges}",
        f"matched/true={format_float(total_matched_selected_edges / max(total_true_edges, 1))}",
        f"exact_graph_assignment_acc={format_float(exact_assignment_acc)}",
    )

    print("\nDegree violations:")
    print(
        "Raw before greedy:",
        f"source_violations={raw_source_violations}",
        f"destination_violations={raw_destination_violations}",
        f"max_source_degree={raw_max_source_degree}",
        f"max_destination_degree={raw_max_destination_degree}",
    )
    print(
        "Matched after greedy:",
        f"source_violations={matched_source_violations}",
        f"destination_violations={matched_destination_violations}",
        f"max_source_degree={matched_max_source_degree}",
        f"max_destination_degree={matched_max_destination_degree}",
    )

    print("\nOverall buffer / length metrics on true links:")
    print(
        f"mae={format_float(reg_overall['mae'])}",
        f"rmse={format_float(reg_overall['rmse'])}",
        f"median_abs_error={format_float(reg_overall['medae'])}",
        f"r2={format_float(reg_overall['r2'])}",
    )

    print("-" * 90)

    return (
        matched_overall["acc"],
        matched_overall["f1"],
        reg_overall["r2"],
    )



# import torch
# import torch.nn.functional as F
#
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     r2_score,
# )
# from collections import defaultdict
# import numpy as np
# import torch
#
#
# def compute_pos_weights(loader, src_configs, device):
#     pos_counts = {}  # sum for all batches
#     total_counts = {}
#
#     for batch in loader:
#         for src_type, dst_list in src_configs.items():
#             for dst_type in dst_list:
#                 edge_type = (src_type, 'predict', dst_type)
#                 if edge_type not in batch.edge_types:
#                     continue
#                 labels = batch[edge_type].edge_value  # tensor[ 0 or 1 ] 1 if the edge is present in the training data
#                 pos = labels.sum().item()
#                 total = labels.numel()
#                 pos_counts[edge_type] = pos_counts.get(edge_type, 0) + pos
#                 total_counts[edge_type] = total_counts.get(edge_type, 0) + total
#
#     pos_weights = {}
#     for edge_type in pos_counts:
#         pos = pos_counts[edge_type]
#         neg = total_counts[edge_type] - pos
#         if pos > 0:
#             w = (neg / pos)**0.5
#         else:
#             w = 1.0
#
#         pos_weights[edge_type] = torch.tensor(w, device=device)
#     return pos_weights
#
#
# # =========================
# # EDGE SCORING
# # =========================
# def compute_edge_scores(model, out, batch, edge_type):
#     src_type, _, dst_type = edge_type
#
#     eidx = batch[edge_type].edge_label_index
#     labels = batch[edge_type].edge_value.float()
#     buffers = batch[edge_type].edge_length.float()
#
#     src = out[src_type][eidx[0]]
#     dst = out[dst_type][eidx[1]]
#
#     pair = torch.cat([src, dst], dim=-1)
#
#     logits = model.link_scorer(pair).squeeze(-1)
#
#     # Positive length prediction
#     pred_buf = F.softplus(model.buf_scorer(pair).squeeze(-1))
#
#     return logits, labels, pred_buf, buffers
#
#
# # =========================
# # TRAIN
# # =========================
# def train(model, train_loader, val_loader, device, src_configs, epochs=20, threshold=0.5, reg_loss_weight=0.01, save_path='../checkpoints/best_hetero_assignment_mlp.pt'):
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#     pos_weights = compute_pos_weights(train_loader, src_configs, device)
#     best_val = 0
#
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0
#
#         stats = {}
#
#         for batch in train_loader:
#             batch = batch.to(device)
#             out = model(batch)
#
#             loss_cls = 0.0
#             loss_reg = 0.0
#
#             for src_type, dst_list in src_configs.items():
#                 for dst_type in dst_list:
#                     edge_type = (src_type, 'predict', dst_type)
#                     if edge_type not in batch.edge_types:
#                         continue
#
#                     logits, labels, pred_buf, buf = compute_edge_scores(model, out, batch, edge_type)
#
#                     if logits.numel() == 0:
#                         continue
#
#                     pos_weight = pos_weights[edge_type]
#
#                     loss_cls_rel = F.binary_cross_entropy_with_logits(
#                         logits,
#                         labels,
#                         pos_weight=pos_weight
#                     )
#                     loss_cls += loss_cls_rel
#
#                     pos_mask = labels == 1
#                     if pos_mask.any():
#                         loss_reg_rel = F.smooth_l1_loss(pred_buf[pos_mask], buf[pos_mask])
#                     else:
#                         loss_reg_rel = logits.new_tensor(0.0)
#                     loss_reg += loss_reg_rel
#
#
#                     # stats
#                     key = str(edge_type)
#                     if key not in stats:
#                         stats[key] = {"loss_cls": 0, "count": 0, "pos": 0, "pred": 0, "loss_reg": 0}
#
#                     stats[key]["loss_cls"] += loss_cls_rel.item()
#                     stats[key]["count"] += 1
#                     stats[key]["pos"] += labels.mean().item()
#                     stats[key]["pred"] += (torch.sigmoid(logits) > threshold).float().mean().item()
#                     stats[key]["loss_reg"] += loss_reg_rel.item()
#
#             loss = loss_cls + loss_reg * reg_loss_weight
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#             total_loss += loss.item()
#
#         # print stats
#         print(f"\nEpoch {epoch} | Loss {total_loss:.4f}")
#         for k, v in stats.items():
#             print(k,
#                   "total_loss=", total_loss,
#                   "loss_cls=", v["loss_cls"]/v["count"],
#                   "pos_rate=", v["pos"]/v["count"],
#                   "pred_pos_rate=", v["pred"]/v["count"],
#                   "loss_reg=", v["loss_reg"]/v["count"])
#
#         val_acc, f1, r2 = evaluate(model, val_loader, device, src_configs, threshold=threshold)
#
#         if f1 > best_val:
#             best_val = f1
#             torch.save(model.state_dict(), save_path)
#             print(f"Saved best model with value {best_val}")
#
#
# # =========================
# # EVAL
# # =========================
# # def evaluate(model, loader, device, src_configs, threshold=0.5):
# #     model.eval()
# #
# #     correct = 0
# #     total = 0
# #
# #     tp, fp, fn = {}, {}, {}
# #
# #     with torch.no_grad():
# #         for batch in loader:
# #             batch = batch.to(device)
# #             out = model(batch)
# #
# #             for src_type, dst_list in src_configs.items():
# #                 for dst_type in dst_list:
# #                     edge_type = (src_type, 'predict', dst_type)
# #                     if edge_type not in batch.edge_types:
# #                         continue
# #
# #                     if edge_type not in tp:
# #                         tp[edge_type] = 0
# #                         fp[edge_type] = 0
# #                         fn[edge_type] = 0
# #
# #                     logits, labels, pred_buf, buf = compute_edge_scores(model, out, batch, edge_type)
# #
# #                     probs = torch.sigmoid(logits)
# #                     preds = (probs > threshold).long()
# #
# #                     tp[edge_type] += ((preds == 1) & (labels == 1)).sum().item()
# #                     fp[edge_type] += ((preds == 1) & (labels == 0)).sum().item()
# #                     fn[edge_type] += ((preds == 0) & (labels == 1)).sum().item()
# #
# #                     correct += (preds == labels.long()).sum().item()
# #                     total += labels.numel()
# #
# #     # ---- Print metrics ----
# #     print(f"Evaluation metrics with threshold {threshold}:")
# #     for key in tp:
# #         tpk, fpk, fnk = tp[key], fp[key], fn[key]
# #
# #         precision = tpk / (tpk + fpk + 1e-8)
# #         recall = tpk / (tpk + fnk + 1e-8)
# #         f1 = 2 * precision * recall / (precision + recall + 1e-8)
# #
# #         print(key,
# #               f"precision={precision:.4f}",
# #               f"recall={recall:.4f}",
# #               f"f1={f1:.4f}")
# #
# #     acc = correct / max(total, 1)
# #     print(f"Overall accuracy: {acc:.4f}")
# #
# #     TP = sum(tp.values())
# #     FP = sum(fp.values())
# #     FN = sum(fn.values())
# #
# #     precision = TP / (TP + FP + 1e-8)
# #     recall = TP / (TP + FN + 1e-8)
# #     f1 = 2 * precision * recall / (precision + recall + 1e-8)
# #     print(f"Overall Precision: {precision:.4f}", f"Recall: {recall:.4f}", f"F1: {f1:.4f}\n")
# #     return acc, f1
#
# def greedy_matching_preds_across_edge_types(
#     edge_label_index_by_type,
#     probs_by_type,
#     min_score=0.5,
# ):
#     """
#     Greedy bipartite matching across multiple edge types.
#
#     Guarantees:
#     - each source node is used at most once
#     - each destination node is used at most once
#     - destination uniqueness is shared across edge types
#
#     Example:
#     ("left",  "predict", "not_scheduled")
#     ("right", "predict", "not_scheduled")
#
#     The same not_scheduled node cannot be selected for both.
#     """
#     preds_by_type = {
#         edge_type: torch.zeros_like(probs, dtype=torch.long)
#         for edge_type, probs in probs_by_type.items()
#     }
#
#     candidates = []
#
#     for edge_type, probs in probs_by_type.items():
#         src_type, _, dst_type = edge_type
#         edge_label_index = edge_label_index_by_type[edge_type]
#
#         probs = probs.view(-1)
#         src_nodes = edge_label_index[0]
#         dst_nodes = edge_label_index[1]
#
#         for i in range(probs.numel()):
#             score = probs[i].item()
#
#             if min_score is not None and score < min_score:
#                 continue
#
#             candidates.append(
#                 (
#                     score,
#                     edge_type,
#                     i,
#                     src_type,
#                     int(src_nodes[i].item()),
#                     dst_type,
#                     int(dst_nodes[i].item()),
#                 )
#             )
#
#     candidates.sort(reverse=True, key=lambda x: x[0])
#
#     used_sources = set()
#     used_destinations = set()
#
#     for _, edge_type, i, src_type, src_id, dst_type, dst_id in candidates:
#         src_key = (src_type, src_id)
#         dst_key = (dst_type, dst_id)
#
#         if src_key in used_sources:
#             continue
#
#         if dst_key in used_destinations:
#             continue
#
#         preds_by_type[edge_type][i] = 1
#         used_sources.add(src_key)
#         used_destinations.add(dst_key)
#
#     return preds_by_type
#
#
# def evaluate(model, loader, device, src_configs, threshold=0.5):
#     model.eval()
#
#     cls_true = defaultdict(list)
#     cls_pred = defaultdict(list)
#
#     buf_true = defaultdict(list)
#     buf_pred = defaultdict(list)
#
#     all_cls_true = []
#     all_cls_pred = []
#
#     all_buf_true = []
#     all_buf_pred = []
#
#     def to_numpy(x):
#         return x.detach().cpu().view(-1).numpy()
#
#     def safe_concat(values):
#         if len(values) == 0:
#             return np.array([])
#         return np.concatenate(values)
#
#     def classification_metrics(y_true, y_pred):
#         if len(y_true) == 0:
#             return float("nan"), float("nan"), float("nan"), float("nan")
#
#         acc = accuracy_score(y_true, y_pred)
#
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             y_true,
#             y_pred,
#             average="binary",
#             zero_division=0,
#         )
#
#         return acc, precision, recall, f1
#
#     def safe_r2(y_true, y_pred):
#         if len(y_true) < 2:
#             return float("nan")
#
#         return r2_score(y_true, y_pred)
#
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             out = model(batch)
#
#             batch_outputs = {}
#             probs_by_type = {}
#             edge_label_index_by_type = {}
#
#             # ------------------------------------------------
#             # 1. Compute probabilities for all edge types first
#             # ------------------------------------------------
#             for src_type, dst_list in src_configs.items():
#                 for dst_type in dst_list:
#                     edge_type = (src_type, "predict", dst_type)
#
#                     if edge_type not in batch.edge_types:
#                         continue
#
#                     logits, labels, pred_buf, buf = compute_edge_scores(
#                         model,
#                         out,
#                         batch,
#                         edge_type,
#                     )
#
#                     labels = labels.view(-1).long()
#                     logits = logits.view(-1)
#                     pred_buf = pred_buf.view(-1)
#                     buf = buf.view(-1)
#
#                     probs = torch.sigmoid(logits)
#
#                     batch_outputs[edge_type] = {
#                         "labels": labels,
#                         "pred_buf": pred_buf,
#                         "buf": buf,
#                     }
#
#                     probs_by_type[edge_type] = probs
#                     edge_label_index_by_type[edge_type] = batch[
#                         edge_type
#                     ].edge_label_index
#
#             # ------------------------------------------------
#             # 2. Greedy matching across all edge types together
#             # ------------------------------------------------
#             preds_by_type = greedy_matching_preds_across_edge_types(
#                 edge_label_index_by_type=edge_label_index_by_type,
#                 probs_by_type=probs_by_type,
#                 min_score=threshold,
#             )
#
#             # ------------------------------------------------
#             # 3. Collect labels/predictions for sklearn metrics
#             # ------------------------------------------------
#             for edge_type, values in batch_outputs.items():
#                 labels = values["labels"]
#                 pred_buf = values["pred_buf"]
#                 buf = values["buf"]
#                 preds = preds_by_type[edge_type]
#
#                 y_true = to_numpy(labels)
#                 y_pred = to_numpy(preds)
#
#                 cls_true[edge_type].append(y_true)
#                 cls_pred[edge_type].append(y_pred)
#
#                 all_cls_true.append(y_true)
#                 all_cls_pred.append(y_pred)
#
#                 # R² only for true existing links
#                 pos_mask = labels == 1
#
#                 if pos_mask.any():
#                     b_true = to_numpy(buf[pos_mask])
#                     b_pred = to_numpy(pred_buf[pos_mask])
#
#                     buf_true[edge_type].append(b_true)
#                     buf_pred[edge_type].append(b_pred)
#
#                     all_buf_true.append(b_true)
#                     all_buf_pred.append(b_pred)
#
#     # ------------------------------------------------
#     # Print per-edge-type metrics
#     # ------------------------------------------------
#     print(f"Evaluation metrics with threshold {threshold}:")
#
#     for edge_type in cls_true:
#         y_true = safe_concat(cls_true[edge_type])
#         y_pred = safe_concat(cls_pred[edge_type])
#
#         acc, precision, recall, f1 = classification_metrics(y_true, y_pred)
#
#         y_buf_true = safe_concat(buf_true[edge_type])
#         y_buf_pred = safe_concat(buf_pred[edge_type])
#
#         r2 = safe_r2(y_buf_true, y_buf_pred)
#
#         print(
#             edge_type,
#             f"accuracy={acc:.4f}",
#             f"precision={precision:.4f}",
#             f"recall={recall:.4f}",
#             f"f1={f1:.4f}",
#             f"r2_buf={r2:.4f}" if not np.isnan(r2) else "r2_buf=nan",
#         )
#
#     # ------------------------------------------------
#     # Print overall metrics
#     # ------------------------------------------------
#     y_true_all = safe_concat(all_cls_true)
#     y_pred_all = safe_concat(all_cls_pred)
#
#     acc, precision, recall, f1 = classification_metrics(
#         y_true_all,
#         y_pred_all,
#     )
#
#     y_buf_true_all = safe_concat(all_buf_true)
#     y_buf_pred_all = safe_concat(all_buf_pred)
#
#     r2_overall = safe_r2(y_buf_true_all, y_buf_pred_all)
#
#     print(f"Overall accuracy: {acc:.4f}")
#
#     print(
#         f"Overall Precision: {precision:.4f}",
#         f"Recall: {recall:.4f}",
#         f"F1: {f1:.4f}",
#         f"R2_buf: {r2_overall:.4f}" if not np.isnan(r2_overall) else "R2_buf: nan",
#         "\n",
#     )
#
#     return acc, f1, r2_overall
