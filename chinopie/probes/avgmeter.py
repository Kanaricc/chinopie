from torch.types import Number


class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg=0
        pass

    def update(self, val: Number, n=1):
        self.val = val
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum/self.cnt
    
    def has_data(self):
        return self.cnt!=0

    def average(self) -> float:
        return self.avg

    def value(self) -> Number:
        return self.val
    
    def reset(self):
        self.val=0
        self.sum=0
        self.cnt=0
        self.avg=0
        
    def __str__(self) -> str:
        return f"{self.name}: {self.value():.5f}(avg {self.average():.5f})"
