
class SeqDataset(Dataset):
    
    def __init__(self, seqs):
        self.seqs = seqs
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq, label = self.seqs[idx]
        return dict(
            seq=torch.Tensor(seq.to_numpy()),
            label = torch.Tensor(label).long()
        )
    
class SeqDataModule(pl.LightningDataModule):
    def __init__(self, train_seqs, test_seqs, batch_size):
        super().__init__()
        self.train_seqs = train_seqs
        self.test_seqs = test_seqs
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = SeqDataset(self.train_seqs)
        self.test_dataset = SeqDataset(self.test_seqs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle= True
        )
    
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle= False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle= False
        )
        