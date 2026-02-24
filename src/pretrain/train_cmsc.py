import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pretrain.baselines_common import (
    MODALITIES,
    MultiModalProjector,
    build_loaders,
    move_batch,
    pairwise_clip_loss,
    save_checkpoint,
    set_seed,
)


def random_crop(signals, crop_len):
    cropped = {}
    for m, x in signals.items():
        length = x.shape[-1]
        if crop_len >= length:
            cropped[m] = x
            continue
        start = torch.randint(0, length - crop_len + 1, (1,), device=x.device).item()
        cropped[m] = x[..., start:start + crop_len]
    return cropped


def evaluate(model, val_loader, device, temperature, crop_ratio):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            crop_len = max(8, int(batch[MODALITIES[0]].shape[-1] * crop_ratio))
            segments = random_crop(batch, crop_len)
            loss = pairwise_clip_loss(model(segments), temperature)
            total += loss.item()
    return total / max(1, len(val_loader))


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    loaders = build_loaders(args.data_path, args.fold_number, args.batch_size, args.val_split, args.seed, args.num_workers)

    sample_batch = next(iter(loaders.train_loader))
    signal_length = max(8, int(sample_batch[MODALITIES[0]].shape[-1] * args.crop_ratio))

    model = MultiModalProjector(signal_length, args.hidden_dim, args.proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run_dir = os.path.join(args.output_path, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_total = 0.0
        for batch in loaders.train_loader:
            batch = move_batch(batch, device)
            crop_len = max(8, int(batch[MODALITIES[0]].shape[-1] * args.crop_ratio))
            segments = random_crop(batch, crop_len)
            loss = pairwise_clip_loss(model(segments), args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        train_loss = train_total / max(1, len(loaders.train_loader))
        val_loss = evaluate(model, loaders.val_loader, device, args.temperature, args.crop_ratio)
        print(f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(run_dir, 'best_ckpt.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMSC-style segment-level contrastive pretraining for WESAD.')
    parser.add_argument('--run_name', type=str, default='cmsc_baseline')
    parser.add_argument('--output_path', type=str, default='./results/pretrain_baselines')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--fold_number', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--crop_ratio', type=float, default=0.5)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
