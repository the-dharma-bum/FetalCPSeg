import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


def init_optimizer(net, config):
    if config.optimizer == 'sgd':
        return optim.SGD(net.parameters(), 
                         lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    if config.optimizer == 'adam':
        return optim.Adam(net.parameters(),     lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adadelta':
        return optim.Adadelta(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adagrad':
        return optim.Adagrad(net.parameters(),  lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adamW':
        return optim.AdamW(net.parameters(),    lr=config.lr, weight_decay=config.weight_decay)

def init_scheduler(optimizer, config):
    if config.scheduler == 'rop':
        return scheduler.ReduceLROnPlateau(optimizer,
                                           mode='min', factor=0.25, patience=20, verbose=False)
    if config.scheduler == 'oneCycle':
        return scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=4*150)