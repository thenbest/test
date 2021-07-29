from model.rmc_trans import Transformer
import datetime
import os
from sklearn.metrics import f1_score
from model.sknet import SKNet101
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import batch_loader, data_split, params
from torch.autograd import Variable

def train(model, train_loader, val_loader, loss, optimizer, n_epoch, model_root, info_step=100):
    last_val_acc = 0

    for epoch in range(n_epoch):
        train_l_sum = 0.0
        train_acc_sum = 0.0
        step = 0

        for step, (X, y) in enumerate(train_loader):
            X = Variable(X).to(device)
#             X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
            y = Variable(y).to(device)
            y_h = model(X)
            print(y_h)
            l = loss(y_h, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            acc = accuracy(y_h, y)
            train_l_sum += l.item()
            train_acc_sum += acc
            if step % info_step == 0:
                print("epoch: %02d, batch_loss: %.3f, batch_acc: %.3f" % (epoch, l.item(), acc))

        lr_scheduler.step()
        train_acc = train_acc_sum / (step + 1)
        val_acc, val_loss = evaluate(model, val_loader, loss)
        print("epoch: %02d, loss: %.3f, acc: %.3f, val_loss: %.3f, val_acc: %.3f" % (
            epoch, train_l_sum / (step + 1), train_acc, val_loss, val_acc))
        if val_acc > last_val_acc:
            last_val_acc = val_acc

        # keep all epoches
        model_name = "epoch_%02d_train_acc_%.3f_val_loss_%.3f_val_acc_%.3f.pth" % (epoch, train_acc, val_loss, val_acc)
        model_path = os.path.join(model_root, model_name)
        torch.save(model.state_dict(), model_path)


def F1(y_h, y):
    return f1_score(y.cpu().numpy(), torch.argmax(y_h, 1).cpu().numpy(), average='weighted')


def accuracy(y_h, y):
    return ((torch.argmax(y_h, 1) == y).sum().float() / y.shape[0]).item()
    # For BCE only
    # y_h_tag = torch.round(torch.sigmoid(y_h))
    # return ((y_h_tag == y).sum().float() / y.shape[0]).item()


def evaluate(model, dataloader, loss):
    # Test: see whether it helps
    model.eval()
    Y_h, Y = [], []
    val_l_sum = 0.0

    with torch.no_grad():
        for step, (X, y) in enumerate(dataloader):
            X = torch.tensor(X, dtype=torch.float32)
            X = X.to(device)
            X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
            y = y.to(device)
            y_h = model(X)
            Y_h.append(y_h)
            Y.append(y)
            l = loss(y_h, y)
            val_l_sum += l.item()

    Y_h = torch.cat(Y_h)
    Y = torch.cat(Y)
    l_val = val_l_sum / (step + 1)

    # Test: See whether it helps
    model.train()
    # return accuracy(Y_h, Y), F1(Y_h, Y)
    return accuracy(Y_h, Y), l_val


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_path = params.train_feats_path
    model_name = "transformer"
    n_epoch = 500
    batch_size = 64
    lr = 0.0001
    lr_decay_step = 5
    lr_decay_gamma = 0.7
    str_trans = "non-standardized"
    if params.use_transform:
        str_trans = "standardized"
    train_name = "test"
    model_root = "models/%s_%s_%dfeatures_%s_%s"
    str_time = datetime.datetime.now().strftime("%m%d%H%M%S")
    model_root = model_root % (str_time, model_name, 128, str_trans, train_name)
    os.mkdir(model_root)
    model = Transformer()
#     if torch.cuda.device_count()>1:
#         print("Let's use ",torch.cuda.device_count(), "GPUs")
#         model = nn.DataParallel(model)
    if torch.cuda.is_available():
        print("CUDA is enable!")
        model = model.to(device)
        model.train()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
#     step_lr = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, lr_decay_gamma)
    if params.use_transform:
        print("use transformation")
        # create a dataset containing all necessary data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(params.means, params.stds)])
        dataset = batch_loader.MelSpecDataset(train_path, transform=transform)
    else:
        print("no transformation")
        dataset = batch_loader.MelSpecDataset(train_path, transform=transforms.ToTensor())
    print(len(dataset))

    split = data_split.DataSplit(dataset, shuffle=True)
    tr_loader, _, val_loader = split.get_split(batch_size=batch_size, num_workers=4)
    train(model, tr_loader, val_loader, loss, optimizer, n_epoch, model_root, info_step=100)

