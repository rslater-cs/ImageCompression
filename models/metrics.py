from torch import nn, Tensor, log10, no_grad

class AverageMetric(nn.Module):
    def __init__(self, metric: nn.Module, dataset_size: int = None, batches: int = None) -> None:
        super().__init__()
        self.metric = metric
        self.current = 0
        self.total = 0.0

        if(dataset_size == None and batches == None):
            raise Exception("Either Batches or Dataset size must be provided")
        elif(dataset_size == None and batches != None):
            self.is_batches = True
            self.size = batches
        elif(dataset_size != None and batches == None):
            self.is_batches = False
            self.size = dataset_size
        else:
            raise Exception("Either Batches or Dataset size must be provided, not both")

    def forward(self, outputs: Tensor, labels: Tensor):
        with no_grad():
            if(self.current == self.size):
                self.current = 0
            loss = self.metric(outputs, labels).item()
            self.total += loss

            if self.is_batches:
                self.current += 1
            else:
                self.current += outputs.shape[0]

            return self.total/self.current
    
class PSNR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs: Tensor, labels: Tensor):
        mse = self.mse(outputs, labels)
        psnr = 10*log10(1.0**2/mse)
        return psnr
