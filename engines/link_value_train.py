import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score


def calculate_edges_values_prediction_loss(
    data,
    model: torch.nn.Module,
    device: torch.device,
    cls_loss,
    reg_loss,
    reg_loss_weight: float = 1.0
):
    """
    Returns:
      total_loss, loss_cls, loss_reg, scores, labels, pos_values_targets, pos_values_pred
    Note: scores/labels contain [pos_scores; neg_scores] and corresponding labels.
          pos_values_targets and pos_values_pred contain only the positive-edge regression targets and predictions.
    """
    data = data.to(device)
    z = model(data.x, data.edge_index, edge_attr=getattr(data, 'edge_attr', None))

    # `edge_predict_index` shape [2, P]; `edge_predict_label` shape [P] boolean/int mask where 1 = positive
    mask = data.edge_predict_label == 1
    pos_edges_all = data.edge_predict_index[:, mask]
    pos_src_idx, pos_dst_idx = pos_edges_all[0].to(device), pos_edges_all[1].to(device)
    neg_edges_all = data.edge_predict_index[:, ~mask]
    neg_src_idx, neg_dst_idx = neg_edges_all[0].to(device), neg_edges_all[1].to(device)

    # Regression targets aligned to positive pairs (shape [P_pos])
    pos_values_targets = data.edge_predict_value[mask].to(device)

    # Score and value predictions from the model
    pos_scores, pos_values = model.score_and_value(z[pos_src_idx], z[pos_dst_idx])
    if neg_src_idx.numel():
        neg_scores, _ = model.score_and_value(z[neg_src_idx], z[neg_dst_idx])
    else:
        neg_scores = torch.empty((0,), device=device)

    # Combined classification arrays (pos then neg)
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([
        torch.ones(pos_scores.size(0), device=device),
        torch.zeros(neg_scores.size(0), device=device)
    ], dim=0)

    # Use provided cls_loss/reg_loss to compute reported losses (their reduction may vary)
    loss_cls = cls_loss(scores, labels)
    loss_reg = reg_loss(pos_values, pos_values_targets)

    total_loss = loss_cls + reg_loss_weight * loss_reg

    # return predicted pos_values as well so validation can compute regression metrics
    return total_loss, loss_cls, loss_reg, scores.detach(), labels.detach(), pos_values_targets.detach(), pos_values.detach()


@torch.no_grad()
def validate_link_value_prediction(
        model,
        val_loader,
        device,
        cls_loss,
        reg_loss,
        reg_loss_weight: float = 1.0
):
    """
    Returns a dict with:
      avg_cls_loss_per_pos: sum(BCE over all samples) / total_num_positives
      auc: ROC AUC over all pairs (pos+neg)
      reg_mse: MSE on positive edges (using predicted pos values)
      reg_r2: R^2 on positive edges
    """
    model.eval()

    total_bce_sum = 0.0        # sum of per-sample BCE (pos+neg) across dataset
    total_pos = 0             # number of positive pairs across dataset

    all_scores = []
    all_labels = []

    all_pos_preds = []        # predicted values for positives
    all_pos_targets = []      # true regression targets for positives

    for data in val_loader:
        # use updated calculate function
        (_, loss_cls, loss_reg, scores, labels, pos_targets, pos_preds) = (
            calculate_edges_values_prediction_loss(
                data, model, device, cls_loss, reg_loss, reg_loss_weight
            )
        )

        # collect scores/labels for AUC
        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

        # collect positive preds/targets for regression metrics
        if pos_targets.numel():
            all_pos_preds.append(pos_preds.cpu())
            all_pos_targets.append(pos_targets.cpu())

        # compute BCE per-sample here (independent of cls_loss.reduction)
        per_sample_bce = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
        total_bce_sum += float(per_sample_bce.sum().item())

        # update pos counter
        total_pos += int(labels.sum().item())

    # assemble arrays
    if all_scores:
        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()
    else:
        all_scores = np.array([])
        all_labels = np.array([])

    # AUC (safe)
    if all_labels.size and len(set(all_labels.tolist())) > 1:
        val_auc = roc_auc_score(all_labels, all_scores)
    else:
        val_auc = float('nan')

    # avg classification loss per positive (sum BCE / total positives)
    if total_pos > 0:
        avg_cls_loss_per_pos = total_bce_sum / total_pos
    else:
        avg_cls_loss_per_pos = float('nan')

    # regression metrics on positives
    if all_pos_preds:
        pos_preds = torch.cat(all_pos_preds).numpy()
        pos_targets = torch.cat(all_pos_targets).numpy()

        # MSE and R2
        mse = float(mean_squared_error(pos_targets, pos_preds))
        r2 = float(r2_score(pos_targets, pos_preds))
    else:
        mse = float('nan')
        r2 = float('nan')

    return {
        'avg_cls_loss_per_pos': float(avg_cls_loss_per_pos),
        'auc': float(val_auc) if not np.isnan(val_auc) else float('nan'),
        'reg_mse': float(mse),
        'reg_r2': float(r2)
    }


def train_link_value_prediction(model, train_loader, val_loader, device, epochs=10, pos_weight=1., lr=1e-3, reg_loss_weight: float = 1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = None if pos_weight is None else torch.tensor([pos_weight])
    cls_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    reg_loss = torch.nn.MSELoss(reduction='mean')  # you can change to L1 if you prefer

    best_val_auc = -float('inf')
    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_pairs = 0
        total_pos_edges = 0

        for data in train_loader:
            total_loss, loss_cls, loss_reg, scores, labels, pos_values_targets, pos_values = (
                calculate_edges_values_prediction_loss(data, model, device, cls_loss, reg_loss, reg_loss_weight))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_cls_loss += float(loss_cls.item()) * labels.size(0)
            total_reg_loss += float(loss_reg.item()) * pos_values_targets.size(0)
            total_pairs += labels.size(0)
            total_pos_edges += pos_values_targets.size(0)

        # epoch metrics (average)
        avg_cls = total_cls_loss / total_pairs if total_pairs > 0 else float('nan')
        avg_reg = total_reg_loss / total_pos_edges if total_pos_edges > 0 else float('nan')

        # Validation
        metrics = validate_link_value_prediction(model, val_loader, device, cls_loss, reg_loss, reg_loss_weight)

        # Save model
        model_name = model.__class__.__name__
        val_auc = metrics['auc']
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'../checkpoints/{model_name}.pth')

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"cls loss: {avg_cls:.6f}, reg loss/pos: {avg_reg:.6f} | "
              f"Val avg_cls_loss_per_pos: {metrics['avg_cls_loss_per_pos']:.6f} | "
              f"Val auc: {metrics['auc']:.6f} | "
              f"Val reg_mse: {metrics['reg_mse']:.4f} | "
              f"Val reg_r2: {metrics['reg_r2']:.4f} | ")

    return model
