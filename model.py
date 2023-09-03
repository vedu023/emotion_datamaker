
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models 
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy  
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
 



class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 7)

        self.accuracy = MulticlassAccuracy(num_classes=7)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
         
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
         
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    
class MyPrintingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

if __name__ == '__main__':

    dataset_path = '/home/linescan/project_X/face_detection_dataset/dataset'
    transform =  transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # print(len(train_set), len(val_set), len(test_set), len(dataset))

    train_loader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=4, shuffle=False)

    # model
    model = LitModel()
    logger = TensorBoardLogger('logs')

    # train model
    early_stop_callback = EarlyStopping(monitor="val_acc")
    trainer = pl.Trainer(logger=logger,min_epochs=5
                         , max_epochs=25, callbacks=[MyPrintingCallback(), early_stop_callback])
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # torch.save(model, 'emotion_detect.pt')

    model = torch.load('/home/linescan/project_X/face_detection_dataset/emotion_detect.pt')
    trainer.test(model, test_loader)


