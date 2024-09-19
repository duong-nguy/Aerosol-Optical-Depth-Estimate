import torch
from torch import nn
import torch.nn.functional as F

import lightning as L



class AODregressor(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs[0], 64, kernel_size=(3, 3),padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3),padding='same')
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3),padding='same')
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3),padding='same')
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3),padding='same')
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3),padding='same')
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3),padding='same')
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3),padding='same')
        self.conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3),padding='same')
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3),padding='same')
        self.pool4 = nn.MaxPool2d(2,2)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dp = nn.Dropout(0.01)

        self.fc = nn.Linear(512,100)
        self.regressor = nn.Linear(100,1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.global_max_pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.relu(x)
        
        x = self.dp(x)
        x = self.regressor(x)
        return x

class LitAODregressor(L.LightningModule):
    def __init__(self,inputs,learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = AODregressor(inputs)

    def training_step(self, batch, batch_idx):
        image, aod = batch
        aod_pred = self.model(image)
        loss_fct = nn.L1Loss()
        metric_fct = PearsonCorrCoef()
        loss = loss_fct(aod_pred,aod)
        metric = metric_fct(aod_pred.cpu(),aod.cpu())
        self.log("MAELoss", loss)
        self.log("Pearson_R", metric)
        return loss

    def validation_step(self, batch, batch_idx):
        image, aod = batch
        with torch.no_grad():
            aod_pred = self.model(image)
            loss_fct = nn.L1Loss()
            metric_fct = PearsonCorrCoef()
            val_loss = loss_fct(aod_pred,aod)
            val_metric= metric_fct(aod_pred.cpu(),aod.cpu())
        self.log("MAELoss", val_loss)
        self.log("Val_Pearson_R", val_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        return {"optimizer": optimizer}

