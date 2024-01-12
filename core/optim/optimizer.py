import torch


# 定义优化器类，包含学习率预热和衰减
class Optimizer:
    def __init__(self, model, lr0, wd, warmup_steps, warmup_start_lr, max_iter, power):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=wd)
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.max_iter = max_iter
        self.power = power
        self.lr0 = lr0
        self.step_num = 0

    def update_lr(self):
        if self.step_num < self.warmup_steps:
            lr = self.warmup_start_lr + (self.lr0 - self.warmup_start_lr) * (
                self.step_num / self.warmup_steps
            )
        else:
            lr = self.lr0 * ((1 - float(self.step_num) / self.max_iter) ** (self.power))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.update_lr()
        self.optimizer.step()
        self.step_num += 1
