import os
import random

from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
from lightning.pytorch.tuner import Tuner

from model import LitAODregressor 
from dataset import AODdataset



def train(args):
    train_ds = AODdataset(args.data_dir,'train',True)
    valid_ds = AODdataset(args.data_dir,'valid',False)
    train_loader = DataLoader(train_ds,batch_size=args.batch_size,num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds,batch_size=args.batch_size,num_workers=args.num_workers)

    image_sample,_ = train_ds[0]
    model = LitAODregressor(image_sample.shape)
    if args.sainity_check:
        trainer = L.Trainer(max_epochs=1,limit_train_batches=5, limit_val_batches=2)
    trainer = L.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model, train_loader,valid_loader)


