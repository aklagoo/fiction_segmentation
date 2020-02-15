import torch


def fit(model, loaders, run, rm, criterion, optimizer, num_epochs, device='cpu'):
    for epoch in range(num_epochs):
        rm.begin_epoch()
        for loader in loaders:
            hidden = model.hidden()
            for X, y in loader:
                # Reshape data
                y_em, y_seg = y[0][1:], y[0][:1]
                x_l, x_c, x_r = X[0].permute(1, 0, 2, 3), X[1], X[2].permute(1, 0, 2, 3)

                # Run model
                pred_em, pred_seg, hidden = model(x_l, x_c, x_r, hidden)

                # Convert segment prediction to single output
                pred_seg = torch.tensor(
                    [torch.argmax(pred_seg, dim=0)]
                ).float().to(device)

                # Calculate loss
                loss_em = criterion(y_em, pred_em).view([1])
                loss_seg = criterion(y_seg, pred_seg).view([1])

                total_loss = sum(loss_em, loss_seg)

                # Perform backprop
                total_loss.backward(retain_graph=True)
                optimizer.step()

                hidden = (x.detach() for x in hidden)

                # Track metrics
                rm.track_metrics(
                    loss_seg,
                    loss_em,
                    pred_seg,
                    y_seg,
                    pred_em,
                    y_em
                )
        rm.end_epoch()
