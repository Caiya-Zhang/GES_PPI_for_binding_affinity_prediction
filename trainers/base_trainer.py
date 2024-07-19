import torch
import wandb
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import KFold


class BaseTrainer:
    @staticmethod
    def transform(args):
        return None

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def train(model, device, loader, optimizer, args, calc_loss, scheduler=None):
        model.train()

        loss_accum = 0
        t = tqdm(loader, desc="Train")
        for step, batch in enumerate(t):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                optimizer.zero_grad()
                pred_list = model(batch)

                loss = calc_loss(pred_list, batch)

                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                detached_loss = loss.item()
                loss_accum += detached_loss
                t.set_description(f"Train (loss = {detached_loss:.4f}, smoothed = {loss_accum / (step + 1):.4f})")
                wandb.log({"train/iter-loss": detached_loss, "train/iter-loss-smoothed": loss_accum / (step + 1)})

        logger.info("Average training loss: {:.4f}".format(loss_accum / (step + 1)))
        return loss_accum / (step + 1)

    @staticmethod
    def cross_validate(model_class, dataset, device, args, calc_loss, n_splits=10):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        cv_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.info(f"Starting fold {fold + 1}/{n_splits}")

            # Initialize model, optimizer, and scheduler for each fold
            model = model_class().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) if args.scheduler else None

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
            val_loader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size)

            BaseTrainer.train(model, device, train_loader, optimizer, args, calc_loss, scheduler)

            # Validate the model
            val_loss = BaseTrainer.validate(model, device, val_loader, calc_loss)
            cv_losses.append(val_loss)

            logger.info(f"Fold {fold + 1}/{n_splits} validation loss: {val_loss:.4f}")

        avg_cv_loss = sum(cv_losses) / n_splits
        logger.info(f"Average cross-validation loss: {avg_cv_loss:.4f}")

        return avg_cv_loss

    @staticmethod
    def validate(model, device, loader, calc_loss):
        model.eval()

        loss_accum = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred_list = model(batch)
                loss = calc_loss(pred_list, batch)
                loss_accum += loss.item()

        return loss_accum / len(loader)

    @staticmethod
    def name(args):
        raise NotImplemented
