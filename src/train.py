"""Train EchoNet on landmark features.

    python train.py --train ../data/train.npz --val ../data/val.npz --epochs 60
Saves models/echonet.pt and models/classes.json
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from augment import random_augment
from config import MODEL_DIR, CLASSES_PATH, SEED, load_vocab
from features import add_velocity
from model import EchoNet


class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, aug):
        self.X, self.y, self.aug = X, y, aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = random_augment(self.X[i]) if self.aug else self.X[i]
        return torch.from_numpy(add_velocity(x)), int(self.y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr, va = np.load(a.train), np.load(a.val)
    vocab = load_vocab()
    classes = vocab or sorted(set(tr["y"].tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    keep_t = np.array([c in idx for c in tr["y"]])
    keep_v = np.array([c in idx for c in va["y"]])
    Xt, yt = tr["X"][keep_t], np.array([idx[c] for c in tr["y"][keep_t]])
    Xv, yv = va["X"][keep_v], np.array([idx[c] for c in va["y"][keep_v]])
    print(f"{len(classes)} classes, {len(Xt)} train / {len(Xv)} val clips")

    dl = torch.utils.data.DataLoader(DS(Xt, yt, True), batch_size=a.bs, shuffle=True, drop_last=len(Xt) > a.bs)
    dv = torch.utils.data.DataLoader(DS(Xv, yv, False), batch_size=128)

    net = EchoNet(len(classes))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = -1.0
    MODEL_DIR.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        net.train()
        for x, y in dl:
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
            sched.step()
        net.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in dv:
                correct += (net(x).argmax(1) == y).sum().item()
                n += len(y)
        acc = correct / max(n, 1)
        print(f"epoch {ep + 1:3d}  val top1 {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(net.state_dict(), MODEL_DIR / "echonet.pt")
    CLASSES_PATH.write_text(json.dumps(classes))
    print("best val top1", best)


if __name__ == "__main__":
    main()
