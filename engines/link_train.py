import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_edges_prediction_loss(data,
                                    model: torch.nn.Module,
                                    device: torch.device,
                                    criterion,
                                    use_dot: bool = False):
    data = data.to(device)
    z = model(data.x, data.edge_index, edge_attr=getattr(data, 'edge_attr', None))

    pos_edges_all = data.edge_predict_index[:, data.edge_predict_label == 1]
    pos_src_idx, pos_dst_idx = pos_edges_all[0].to(device), pos_edges_all[1].to(device)
    neg_edges_all = data.edge_predict_index[:, data.edge_predict_label == 0]
    neg_src_idx, neg_dst_idx = neg_edges_all[0].to(device), neg_edges_all[1].to(device)

    if use_dot:
        pos_scores = (z[pos_src_idx] * z[pos_dst_idx]).sum(dim=1)
        neg_scores = (z[neg_src_idx] * z[neg_dst_idx]).sum(dim=1) if neg_src_idx.numel() else torch.empty((0,),
                                                                                                          device=device)
    else:
        pos_scores = model.score_edges(z[pos_src_idx], z[pos_dst_idx])
        neg_scores = model.score_edges(z[neg_src_idx], z[neg_dst_idx]) if neg_src_idx.numel() else torch.empty((0,),
                                                                                                               device=device)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([torch.ones(pos_scores.size(0), device=device),
                        torch.zeros(neg_scores.size(0), device=device)], dim=0)

    loss = criterion(scores, labels)
    return loss, scores, labels


def evaluate_link_prediction(model, loader, device, use_dot=False):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    total_loss, n_pairs = 0.0, 0
    all_scores, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            loss, scores, labels = calculate_edges_prediction_loss(data, model, device, criterion, use_dot)

            total_loss += float(loss.item()) * labels.size(0)
            n_pairs += labels.size(0)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_pairs if n_pairs > 0 else float('nan')
    auc = roc_auc_score(all_labels, all_scores) if n_pairs > 0 else float('nan')
    ap = average_precision_score(all_labels, all_scores) if n_pairs > 0 else float('nan')

    return avg_loss, auc, ap


def train_link_prediction(model, train_loader, val_loader, device, epochs=10, pos_weight=1., lr=1e-3, use_dot=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = None if pos_weight is None else torch.tensor([pos_weight])
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    best_val_auc = -float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss, n_pairs = 0.0, 0

        for data in train_loader:
            loss, scores, labels = calculate_edges_prediction_loss(data, model, device, criterion, use_dot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            n_pairs += labels.size(0)

        avg_train_loss = total_loss / n_pairs if n_pairs > 0 else float('nan')

        # Validation
        val_loss, val_auc, val_ap = evaluate_link_prediction(model, val_loader, device, use_dot)

        # Save model
        model_name = model.__class__.__name__
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'../checkpoints/{model_name}.pth')

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val AP: {val_ap:.4f}")

    return model

