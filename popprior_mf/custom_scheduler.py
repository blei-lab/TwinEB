"""
A training module with custom scheduler.
"""

def train_model(
    model,
    dataset,
    batch_size,
    max_steps,
    learning_rate,
    device,
    log_cadence,
    tolerance,
    summary_writer,
    param_save_dir,
    retrain=False,
):
    """
    Train the model using the Adam optimizer and a scheduler.
    """
    # if 'PMF' not in model.__class__.__name__:
    #     init_params(model, dataset)

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    iterator = data_loader.__iter__()
    optimizers, schedulers = _setup_optimization(
        model, dataset, batch_size, learning_rate
    )

    # assert learning_rate == 1e-10
    # optimizer_rows = torch.optim.SGD([{"params": model.row_distribution.parameters()}], lr=learning_rate, momentum=.9)
    # optimizer_cols = torch.optim.Adam([{"params": model.column_distribution.parameters()}], lr=1e-1,)
    # scheduler_rows = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_rows, "min", factor=0.5, patience=5, verbose=True)
    # scheduler_cols = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cols, "min", factor=0.5, patience=5, verbose=True)
    # optimizers = [optimizer_rows, optimizer_cols]
    # schedulers = [scheduler_rows, scheduler_cols]

    schedule_free_epochs = 20
    schedule_free_epochs_indx = 0
    start_time = time.time()
    prev_loss = 1e8
    tol = 0  # tolerance counter
    epoch_len = np.round(dataset.num_datapoints / batch_size)
    print(f"Epoch length: {epoch_len}")
    print(f"DataLoader len = {len(data_loader)}")
    epoch = 0
    start_scheduler = False
    # Keep track of original learning rates and set them to smaller values
    orig_lrs = []
    for param_group in optimizers[0].param_groups:
        orig_lrs.append(param_group['lr'])
        #param_group['lr'] = param_group['lr']*0.1
    optimizers[0].param_groups[0]['lr'] = optimizers[0].param_groups[0]['lr']*0.1

    for step in range(max_steps):
        try:
            datapoints_indices, x_train, holdout_mask = iterator.next()
        except StopIteration:
            iterator = data_loader.__iter__()
            datapoints_indices, x_train, holdout_mask = iterator.next()

        datapoints_indices = datapoints_indices.to(device)
        x_train = x_train.to(device)
        for optim in optimizers:
            optim.zero_grad()
        elbo = model(datapoints_indices, x_train, holdout_mask, step)
        loss = -elbo
        loss.backward(retain_graph=True)
        for optim in optimizers:
            optim.step()

        if step % epoch_len == 0:
            epoch += 1

        # Start small (1/10th) for 10 epochs
        # Then start increasing until the original lr is reached
        if epoch > 10 and not start_scheduler:
            if step % epoch_len == 0:
                for i in range(len(optimizers[0].param_groups)):
                    optimizers[0].param_groups[i]['lr'] = np.minimum(orig_lrs[i], orig_lrs[i] * 1.05)
                if optimizers[0].param_groups[i]['lr'] == orig_lrs[i]:
                    start_scheduler = True
                    print("Starting scheduler")
        
        if start_scheduler:
            schedule_free_epochs_indx += 1

        if start_scheduler and step % epoch_len == 0 and schedule_free_epochs_indx > schedule_free_epochs:
            for scheduler in schedulers:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(loss)
                else:
                    scheduler.step()

        # for scheduler in schedulers:
        #     if scheduler is not None and epoch > schedule_free_epochs:
        #         if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        #             scheduler.step(loss)
        #         else:
        #             scheduler.step()

        # for scheduler in schedulers:
        #     scheduler.step(epoch + step / len(data_loader))

        if step == 0 or step % log_cadence == log_cadence - 1:
            duration = (time.time() - start_time) / (step + 1)
            lr_str = ''
            # for scheduler in schedulers:
            #     if scheduler is not None:
            #         for i in scheduler.optimizer.param_groups:
            #             lr_str += f'{i["lr"]:.3f}_'
            #     else:
            #         lr_str += f'{learning_rate:.3f}_'
            for param_group in optimizers[0].param_groups:
                lr_str += f"{param_group['lr']:.3f} "

            print(
                "Step: {:>3d} Epoch {:>3d}, ELBO: {:.3f} ({:.3f} sec) LR: {}".format(
                    step + 1, epoch, -loss, duration, lr_str
                )
            )
            summary_writer.add_scalar("loss", loss, step)

            if loss < prev_loss:
                tol = 0
                prev_loss = loss
            else:
                tol += 1
                prev_loss = loss
            if step == max_steps - 1 or tol >= tolerance:
                if tol >= tolerance:
                    print(f"Loss goes up for {tol} consecutive prints. Stop training.")
                    break

                if step == max_steps - 1:
                    print("Maximum step reached. Stop training.")
    if ~retrain:
        model.save_txt(param_save_dir, train_data=dataset.train)
        torch.save(model.state_dict(), os.path.join(param_save_dir, "model_trained.pt"))

    return model

