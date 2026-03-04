import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, average_precision_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

# Initialize SummaryWriter
writer = SummaryWriter('../run_scalars/link_value_model')


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
    model.to(device)
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

    # print("scores device:", scores.device)
    # print("labels device:", labels.device)

    # Use provided cls_loss/reg_loss to compute reported losses (their reduction may vary)
    loss_cls = cls_loss(scores, labels)
    # TODO: if BCE with 'sum'
    # loss_cls_raw = cls_loss(scores, labels)
    # loss_cls = loss_cls_raw / max(1.0, pos_scores.size(0))
    ########
    loss_reg = reg_loss(pos_values, pos_values_targets)

    total_loss = loss_cls + reg_loss_weight * loss_reg

    # return predicted pos_values as well so validation can compute regression metrics
    return total_loss, loss_cls, loss_reg, scores.detach(), labels.detach(), pos_values_targets.detach(), pos_values.detach()


@torch.no_grad()
def validate_diagnostics(model, val_loader, device, cls_loss, reg_loss):
    model.eval()
    all_scores = []
    all_labels = []
    pos_preds = []
    pos_targets = []
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_pairs = 0
    total_pos_edges = 0
    metrics = dict()

    with (((torch.no_grad()))):
        for data in val_loader:
            total_loss, loss_cls, loss_reg, scores, labels, pos_targets_batch, pos_preds_batch = calculate_edges_values_prediction_loss(data, model, device, cls_loss, reg_loss)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            if pos_preds_batch.numel():
                pos_preds.append(pos_preds_batch.cpu())
                pos_targets.append(pos_targets_batch.cpu())

            total_cls_loss += float(loss_cls.item()) * labels.size(0)
            total_reg_loss += float(loss_reg.item()) * pos_targets_batch.size(0)
            total_pairs += labels.size(0)
            total_pos_edges += pos_targets_batch.size(0)

            # epoch metrics (average)
        avg_cls = total_cls_loss / total_pairs if total_pairs > 0 else float('nan')
        avg_reg = total_reg_loss / total_pos_edges if total_pos_edges > 0 else float('nan')
        metrics['avg_cls_loss'] = avg_cls
        metrics['avg_reg_loss'] = avg_reg


    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # mean predicted probabilities
    probs = 1 / (1 + np.exp(-all_scores))
    mean_pos_prob = probs[all_labels==1].mean() if (all_labels==1).any() else np.nan
    mean_neg_prob = probs[all_labels==0].mean() if (all_labels==0).any() else np.nan

    ap = average_precision_score(all_labels, all_scores) if len(set(all_labels))>1 else np.nan
    # precision-recall top checks
    prec, rec, thr = precision_recall_curve(all_labels, all_scores)

    metrics['auc'] = roc_auc_score(all_labels, all_scores) if len(set(all_labels))>1 else np.nan
    metrics['ap'] = ap  # average precision
    metrics['mean_pos_prob'] = mean_pos_prob
    metrics['mean_neg_prob'] = mean_neg_prob
    metrics['precision@0.45'] = np.max(prec[rec>=0.45]) if (rec>=0.45).any() else np.nan

    if pos_preds:
        pos_preds = torch.cat(pos_preds).numpy()
        pos_targets = torch.cat(pos_targets).numpy()
        from sklearn.metrics import mean_squared_error, r2_score
        metrics['mse'] = mean_squared_error(pos_targets, pos_preds)
        metrics['r2'] = r2_score(pos_targets, pos_preds)

    return metrics


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
    pos_weight_tensor = torch.as_tensor(pos_weight, device=device)
    cls_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_tensor)
    # cls_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    reg_loss = torch.nn.MSELoss(reduction='mean')  # you can change to L1 if you prefer

    best_val_auc = -float('inf')
    for epoch in range(epochs):
        model.train()
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_pairs = 0
        total_pos_edges = 0

        for i, data in enumerate(train_loader):
            total_loss, loss_cls, loss_reg, scores, labels, pos_values_targets, pos_values = (
                calculate_edges_values_prediction_loss(data, model, device, cls_loss, reg_loss, reg_loss_weight))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Write to TensorBoard:
            writer.add_scalar('loss/cls_loss', loss_cls, epoch * len(train_loader) + i)
            writer.add_scalar('loss/reg_loss', loss_reg, epoch * len(train_loader) + i)

            total_cls_loss += float(loss_cls.item()) * labels.size(0)
            total_reg_loss += float(loss_reg.item()) * pos_values_targets.size(0)
            total_pairs += labels.size(0)
            total_pos_edges += pos_values_targets.size(0)

        # epoch metrics (average)
        avg_cls = total_cls_loss / total_pairs if total_pairs > 0 else float('nan')
        avg_reg = total_reg_loss / total_pos_edges if total_pos_edges > 0 else float('nan')
        writer.add_scalar('avg_loss/avg_cls_loss', avg_cls, epoch)
        writer.add_scalar('avg_loss/avg_reg_loss', avg_reg, epoch)

        # Validation
        # metrics = validate_link_value_prediction(model, val_loader, device, cls_loss, reg_loss, reg_loss_weight)
        metrics = validate_diagnostics(model, val_loader, device, cls_loss, reg_loss)
        writer.add_scalar('avg_loss/val_avg_cls_loss', metrics['avg_cls_loss'], epoch)
        writer.add_scalar('avg_loss/val_avg_reg_loss', metrics['avg_reg_loss'], epoch)
        writer.add_scalar('val/auc', metrics['auc'], epoch)
        writer.add_scalar('val/ap', metrics['ap'], epoch)
        writer.add_scalar('val/mean_pos_prob', metrics['mean_pos_prob'], epoch)
        writer.add_scalar('val/mean_neg_prob', metrics['mean_neg_prob'], epoch)
        writer.add_scalar('val/precision@0.45', metrics['precision@0.45'], epoch)
        writer.add_scalar('val/mse', metrics['mse'], epoch)
        writer.add_scalar('val/r2', metrics['r2'], epoch)

        # Save model
        model_name = model.__class__.__name__
        val_auc = metrics['auc']
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_name': model_name,
                'config':  model.config,
                'pos_weight': pos_weight,
                'reg_loss_weight': reg_loss_weight,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_val_auc,
                'epoch': epoch,
            }, f'../checkpoints/{model_name}.pth')
            # torch.save(model.state_dict(), f'../checkpoints/{model_name}.pth')

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"cls avg_loss: {avg_cls:.4f}, reg avg_loss: {avg_reg:.4f} || "
              f"Val: auc: {metrics['auc']:.4f} | "
              f"ap: {metrics['ap']:.4f} | "
              f"mean_pos_prob: {metrics['mean_pos_prob']:.4f} | "
              f"mean_neg_prob: {metrics['mean_neg_prob']:.4f} | "
              f"precision@0.45: {metrics['precision@0.45']:.4f} | "
              f"mse: {metrics['mse']:.4f} | "
              f"r2: {metrics['r2']:.4f} | ")

    return model
