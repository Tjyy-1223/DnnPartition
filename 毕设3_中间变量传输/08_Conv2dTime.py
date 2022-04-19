import torch
import function
import torch.nn as nn




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1,64,112,112))
    x = x.to(device)

