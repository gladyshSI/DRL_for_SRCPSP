import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, average_precision_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def assignment_oracle(src, dst, theta):
    """
    Maximum-weight bipartite matching with optional IDLE decisions.

    Parameters
    ----------
    src : Tensor [P]
        Machine-node IDs.

    dst : Tensor [P]
        Candidate-job IDs.

    theta : Tensor [P]
        Scores of real machine-job assignments.

    Returns
    -------
    y_hat : Tensor [P]
        Incidence vector for REAL assignments only.
        Dummy assignments are implicit.
    """

    assert src.ndim == 1
    assert dst.ndim == 1
    assert theta.ndim == 1
    assert src.shape == dst.shape == theta.shape

    p = theta.numel()
    assert p > 0

    device = theta.device
    dtype = theta.dtype

    src_list = src.detach().cpu().tolist()
    dst_list = dst.detach().cpu().tolist()
    theta_np = theta.detach().cpu().numpy()

    # Unique machines/jobs, preserving order
    sources = list(dict.fromkeys(src_list))
    destinations = list(dict.fromkeys(dst_list))

    n_machines = len(sources)
    n_jobs = len(destinations)

    src_to_row = {s: i for i, s in enumerate(sources)}
    dst_to_col = {d: j for j, d in enumerate(destinations)}

    # For now we assume ALL machine-job combinations are present
    assert p == n_machines * n_jobs, (
        "Oracle expects the complete machine × candidate bipartite graph"
    )

    # Real jobs + one dummy column per machine
    #
    # First n_jobs columns = real jobs
    # Last n_machines columns = dummy jobs
    T = np.zeros(
        (n_machines, n_jobs + n_machines),
        dtype=np.float64
    )

    edge_to_k = {}

    # Fill scores of real assignments
    for k, (s, d, score) in enumerate(
            zip(src_list, dst_list, theta_np)
    ):
        pair = (s, d)

        if pair in edge_to_k:
            raise ValueError(f"Duplicate edge {pair}")

        i = src_to_row[s]
        j = dst_to_col[d]

        T[i, j] = float(score)
        edge_to_k[pair] = k

    # Dummy columns already have score 0.
    #
    # Old SciPy:
    # maximizing T == minimizing -T
    row_idx, col_idx = linear_sum_assignment(-T)

    # IMPORTANT: output only has dimensions of REAL edges
    y_hat = torch.zeros(
        p,
        dtype=dtype,
        device=device
    )

    for i, j in zip(row_idx, col_idx):

        # Dummy assignment -> machine stays idle this step
        if j >= n_jobs:
            continue

        s = sources[i]
        d = destinations[j]

        k = edge_to_k[(s, d)]
        y_hat[k] = 1.0

    return y_hat


@torch.no_grad()
def perturbed_assignment_oracle(
        src,
        dst,
        theta,
        epsilon=0.1,
        n_samples=16
):
    y_sum = torch.zeros_like(theta)

    for _ in range(n_samples):
        noise = torch.randn_like(theta)

        y_k = assignment_oracle(
            src,
            dst,
            theta + epsilon * noise
        )

        y_sum += y_k

    return y_sum / n_samples


def calculate_edges_values_prediction_loss(
        data,
        model,
        device,
        reg_loss,
        reg_loss_weight=1.0,
        epsilon=0.1,
        n_perturbations=16
):
    data = data.to(device)

    z = model(
        data.x,
        data.edge_index,
        edge_attr=getattr(data, 'edge_attr', None)
    )

    src = data.edge_predict_index[0]
    dst = data.edge_predict_index[1]

    theta, buffer_pred = model.score_and_value(
        z[src],
        z[dst]
    )

    y_target = data.edge_predict_label.float()

    assert theta.shape == y_target.shape

    # Which scheduling slice does each candidate edge belong to?
    pair_batch = data.batch[src]

    y_smooth = torch.zeros_like(theta)

    with torch.no_grad():
        for g in pair_batch.unique():
            mask_g = pair_batch == g

            y_smooth[mask_g] = perturbed_assignment_oracle(
                src[mask_g],
                dst[mask_g],
                theta[mask_g],
                epsilon=epsilon,
                n_samples=n_perturbations
            )

    grad_theta = y_smooth - y_target

    n_graphs = pair_batch.unique().numel()

    loss_struct = torch.sum(
        theta * grad_theta.detach()
    ) / n_graphs

    # Buffer regression
    positive_mask = y_target.bool()

    if positive_mask.any():
        buffer_target = data.edge_predict_value[
            positive_mask
        ]

        loss_reg = reg_loss(
            buffer_pred[positive_mask],
            buffer_target
        )
    else:
        loss_reg = buffer_pred.sum() * 0.0

    total_loss = (
            loss_struct
            + reg_loss_weight * loss_reg
    )

    return (
        total_loss,
        loss_struct,
        loss_reg,
        theta.detach(),
        y_smooth.detach(),
        y_target.detach(),
        buffer_pred.detach(),
        positive_mask.detach()
    )


@torch.no_grad()
def validate_diagnostics(model, val_loader, device, reg_loss, reg_loss_weight):
    model.eval()
    all_scores = []
    all_labels = []
    pos_preds = []
    pos_targets = []
    total_loss_struct = 0.0
    total_loss_reg = 0.0
    total_pairs = 0
    total_pos_edges = 0
    metrics = dict()

    with torch.no_grad():
        for data in val_loader:
            total_loss, loss_struct, loss_reg, theta, y_smooth, y_target, buffer_pred, positive_mask = (
                calculate_edges_values_prediction_loss(
                    data,
                    model,
                    device,
                    reg_loss,
                    reg_loss_weight=reg_loss_weight,
                    epsilon=0.1,
                    n_perturbations=16
                ))

            ############################################
            fractional = (
                    (y_smooth > 0.0) &
                    (y_smooth < 1.0)
            ).float().mean().item()

            print(
                f"theta: min={theta.min().item():.3f}, "
                f"mean={theta.mean().item():.3f}, "
                f"max={theta.max().item():.3f}, "
                f"fractional_y={fractional:.3f}"
            )
            ###########################################

            all_scores.append(theta.cpu())
            all_labels.append(y_target.cpu())

            n_pos = int(positive_mask.sum().item())
            if n_pos > 0:
                pos_preds.append(buffer_pred[positive_mask].cpu())
                pos_targets.append(data.edge_predict_value[positive_mask].cpu())

            n_pairs = theta.numel()
            total_loss_struct += float(loss_struct.item()) * n_pairs
            total_loss_reg += float(loss_reg.item()) * n_pos
            total_pairs += n_pairs
            total_pos_edges += n_pos

        # epoch metrics (average)
        avg_struct = total_loss_struct / total_pairs if total_pairs > 0 else float('nan')
        avg_reg = total_loss_reg / total_pos_edges if total_pos_edges > 0 else float('nan')
        metrics['avg_loss_struct'] = avg_struct
        metrics['avg_loss_reg'] = avg_reg

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


def train_link_value_coaml_prediction(model, train_loader, val_loader, device, epochs=10, pos_weight=1., lr=1e-3,
                                reg_loss_weight: float = 1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_loss = torch.nn.MSELoss(reduction='mean')  # you can change to L1 if you prefer

    # Initialize SummaryWriter
    writer = SummaryWriter('../run_scalars/link_value_coaml_model')

    best_val_auc = -float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss_struct = 0.0
        total_loss_reg = 0.0
        total_pairs = 0
        total_pos_edges = 0

        for i, data in enumerate(train_loader):
            total_loss, loss_struct, loss_reg, theta, y_smooth, y_target, buffer_pred, positive_mask = (
                calculate_edges_values_prediction_loss(
                    data,
                    model,
                    device,
                    reg_loss,
                    reg_loss_weight=reg_loss_weight,
                    epsilon=0.1,
                    n_perturbations=16
                ))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Write to TensorBoard:
            writer.add_scalar('loss/loss_struct', loss_struct, epoch * len(train_loader) + i)
            writer.add_scalar('loss/loss_reg', loss_reg, epoch * len(train_loader) + i)

            n_pairs = theta.numel()
            n_pos = int(positive_mask.sum().item())
            total_loss_struct += float(loss_struct.item()) * n_pairs
            total_loss_reg += float(loss_reg.item()) * n_pos
            total_pairs += n_pairs
            total_pos_edges += n_pos

        # epoch metrics (average)
        avg_struct = total_loss_struct / total_pairs if total_pairs > 0 else float('nan')
        avg_reg = total_loss_reg / total_pos_edges if total_pos_edges > 0 else float('nan')
        writer.add_scalar('avg_loss/avg_loss_struct', avg_struct, epoch)
        writer.add_scalar('avg_loss/avg_loss_reg', avg_reg, epoch)

        # Validation
        # metrics = validate_link_value_prediction(model, val_loader, device, cls_loss, reg_loss, reg_loss_weight)
        metrics = validate_diagnostics(model, val_loader, device, reg_loss, reg_loss_weight)
        writer.add_scalar('avg_loss/val_avg_loss_struct', metrics['avg_loss_struct'], epoch)
        writer.add_scalar('avg_loss/val_avg_loss_reg', metrics['avg_loss_reg'], epoch)
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
                'config': model.config,
                'pos_weight': pos_weight,
                'reg_loss_weight': reg_loss_weight,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_val_auc,
                'epoch': epoch,
            }, f'../checkpoints/{model_name}_Coaml.pth')
            # torch.save(model.state_dict(), f'../checkpoints/{model_name}.pth')

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"struct avg_loss: {avg_struct:.4f}, reg avg_loss: {avg_reg:.4f} || "
              f"Val: auc: {metrics['auc']:.4f} | "
              f"ap: {metrics['ap']:.4f} | "
              f"mean_pos_prob: {metrics['mean_pos_prob']:.4f} | "
              f"mean_neg_prob: {metrics['mean_neg_prob']:.4f} | "
              f"precision@0.45: {metrics['precision@0.45']:.4f} | "
              f"mse: {metrics['mse']:.4f} | "
              f"r2: {metrics['r2']:.4f} | ")


    return model
