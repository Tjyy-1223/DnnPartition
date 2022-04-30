import function
import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)

    modelIndex = 1
    model = function.getDnnModel(modelIndex)
    model.to(device)

    function.warmUpGpu(model,x,device)
    x,mytime = function.recordTimeGpu(model,x,device,300)
    print(mytime)