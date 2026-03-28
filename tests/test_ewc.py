import torch
import torch.nn as nn
from conti.training.ewc import EWCPenalty


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_init():
    m = _Tiny()
    ewc = EWCPenalty(m, lambda_ewc=1000.0)
    assert ewc.lambda_ewc == 1000.0
    assert len(ewc._ref_params) > 0


def test_zero_at_init():
    m = _Tiny()
    ewc = EWCPenalty(m, lambda_ewc=1000.0)
    # fake fisher
    for name, param in m.named_parameters():
        if param.requires_grad:
            ewc._fisher_diag[name] = torch.ones_like(param.data)

    pen = ewc.penalty(m)
    assert abs(float(pen)) < 1e-5


def test_increases():
    m = _Tiny()
    ewc = EWCPenalty(m, lambda_ewc=1000.0)
    for name, param in m.named_parameters():
        if param.requires_grad:
            ewc._fisher_diag[name] = torch.ones_like(param.data)

    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)

    pen = ewc.penalty(m)
    assert float(pen) > 0


def test_ready():
    m = _Tiny()
    ewc = EWCPenalty(m)
    assert not ewc.is_ready
    for name, param in m.named_parameters():
        if param.requires_grad:
            ewc._fisher_diag[name] = torch.ones_like(param.data)
    assert ewc.is_ready


def test_lambda_scales():
    m = _Tiny()
    lo = EWCPenalty(m, lambda_ewc=1.0)
    hi = EWCPenalty(m, lambda_ewc=10000.0)
    for name, param in m.named_parameters():
        if param.requires_grad:
            lo._fisher_diag[name] = torch.ones_like(param.data)
            hi._fisher_diag[name] = torch.ones_like(param.data)

    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)

    assert float(hi.penalty(m)) > float(lo.penalty(m))
