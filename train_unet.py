import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models.unet import UNet
import torch.optim as optim

class SliceDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)/255.0
        mask = (img > img.mean() + 0.25).astype(np.float32)
        img = img[np.newaxis,...]
        mask = mask[np.newaxis,...]
        return torch.tensor(img), torch.tensor(mask)

def train(folder, epochs=3, out="models/unet_best.pt"):
    ds = SliceDataset(folder)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    model = UNet(in_channels=1, out_channels=1)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        total=0
        model.train()
        for x,y in dl:
            outp = model(x)
            loss = loss_fn(outp, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}, loss {total/len(dl):.4f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(model.state_dict(), out)
    print("Saved model to", out)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='examples/synthetic_phantom')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--out', default='models/unet_best.pt')
    args = p.parse_args()
    train(args.data, epochs=args.epochs, out=args.out)
