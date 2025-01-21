from model.pyramid_resnet import LocalActivationResNet
from model.building_blocks import ProjectionLayer, AMIL
from losses.loss import CoxLoss
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



