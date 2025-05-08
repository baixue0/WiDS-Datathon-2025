import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, meta_dim, hidden_dim, output_dim, dropout):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim*8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*8, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*8),
            nn.ReLU(),
            nn.Linear(hidden_dim*8, input_dim),
        )

        # MLP
        self.mlp = nn.Sequential(
            #nn.Linear(hidden_dim+meta_dim, hidden_dim+meta_dim),
            nn.Linear(hidden_dim+meta_dim, output_dim),
        )
        
        # Apply weights initialization
        self.apply(self.weights_init_uniform_rule)
        
    def forward(self, x, meta):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, self.mlp(torch.cat([encoded, meta], dim=1)), decoded
    
    def weights_init_uniform_rule(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def train_model(self, epoch, trainloader, optimizer, loss_mse, loss_bce, device):
        self.train()
        train_loss = 0
        for batch in trainloader:
            x = batch[0].float().to(device)
            meta = batch[2].float().to(device)
            ecd, cl, recon = self(x, meta)
            y = batch[1].float().to(device)
            l1_norm = sum(p.abs().sum() for p in self.mlp.parameters())
            loss = loss_mse(recon, x) + loss_bce(cl.squeeze(), y) * 50 + l1_norm * 0.01
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(trainloader.dataset)
        return train_loss

    def validate_model(self, epoch, valloader, device):
        self.eval()
        encoded, xs, recs, target, metas, clses = [], [], [], [], [], []
        with torch.no_grad():
            for batch in valloader:
                x = batch[0].float().to(device)
                meta = batch[2].float().to(device)
                ecd, cl, recon = self(x, meta)
                target.append(batch[1].cpu().numpy())
                xs.append(x.cpu().numpy())
                recs.append(recon.cpu().numpy())
                encoded.append(ecd.cpu().numpy())
                metas.append(batch[2].numpy())
                clses.append(torch.sigmoid(cl).cpu().numpy())
        encoded = np.vstack(encoded)
        xs = np.vstack(xs)
        recs = np.vstack(recs)
        target = np.hstack(target)
        metas = np.vstack(metas)
        clses = np.vstack(clses)
        return encoded, xs, recs, target, metas, clses

    def predict(self, test_loader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].float().to(device)
                meta = batch[1].float().to(device)
                ecd,cls,recon = self(x,meta)
                pred = cls.detach().cpu().numpy()
                pred = (pred>0.5).astype(int)
                preds += pred.tolist()
        return np.array(preds)