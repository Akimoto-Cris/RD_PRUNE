import torch
import numpy as np
import torch.nn.functional as F


def trainer_loader(args):
    return train

def initialize_weight(model,loader):
    batch = next(iter(loader))
    device = next(model.parameters()).device
    with torch.no_grad():
        model(batch[0].to(device))


def train(model,optpack,train_loader,test_loader,print_steps=-1,log_results=False,log_path='log.txt'):
    model.train()
    opt = optpack["optimizer"](model.parameters())
    if optpack["scheduler"] is not None:
        sched = optpack["scheduler"](opt)
    else:
        sched = None
    num_steps = optpack["steps"]
    device = next(model.parameters()).device
    
    results_log = []
    training_step = 0
    best_acc = 0.
    best_state_dict = None
    
    test_acc, test_loss = test(model,test_loader)
    from tools.utils import get_model_sparsity
    print(f'Before train: test acc: {test_acc:.2f}, sparsity: {get_model_sparsity(model)}')


    while True and num_steps > 0:
        for i,data in enumerate(train_loader):
            opt.zero_grad()
            try:
                x, y = data
                x = x.to(device)
                y = y.to(device)
            except:
                x = data[0]["data"]
                y = data[0]["label"].long()[:, 0]
                
            training_step += 1
    
            yhat = model(x)
            loss = F.cross_entropy(yhat, y)
            celoss = loss.clone().detach()

            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            
            if print_steps != -1 and training_step%print_steps == 0:
                print(f"\rSteps: {training_step}/{num_steps} trainloss: {celoss}", end="")

            if training_step >= num_steps:
                if hasattr(train_loader, "reset"):
                    train_loader.reset()
                break
        if hasattr(train_loader, "reset"):
            train_loader.reset()
        if training_step >= num_steps:
            break

    train_acc,train_loss = -1, -1
    test_acc,test_loss      = test(model,test_loader)
    print(f'\nAfter train: Test acc: {test_acc:.2f}')
    return [test_acc,test_loss,train_acc,train_loss]

def test(model,loader):
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    loss    = 0
    total   = 0
    for i,data in enumerate(loader):
        try:
            x, y = data
            x = x.to(device)
            y = y.to(device)
        except:
            x = data[0]["data"]
            y = data[0]["label"].long()[:, 0]
        with torch.no_grad():
            yhat    = model(x)
            _,pred  = yhat.max(1)
        correct += pred.eq(y).sum().item()
        loss += F.cross_entropy(yhat,y)*len(x)
        total += len(x)
    
    if hasattr(loader, "reset"):
        loader.reset()

    acc     = correct/total * 100.0
    loss    = loss/total
    
    model.train()
    
    return acc,loss

