import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from collections import abc, OrderedDict
import torch.nn.functional as F
import time
from torchvision import models

models.mobilenet_v2()