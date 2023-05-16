import csv

## Easy plotting for loss and psnr progress
class MetricLogger:
    
    # overwites a file unless specified to continue a checkpoint model
    def __init__(self, path, name, mode='w'):
        self.path = f'{path}/{name}.csv'
        file = open(self.path, mode, newline='')
        writer = csv.writer(file)
        if(mode=='w'):
            writer.writerow(["epoch", f"{name}_loss", f"{name}_psnr"])

    def put(self, epoch, avg_loss, avg_psnr):
        file = open(self.path, 'a', newline='')
        writer = csv.writer(file)

        writer.writerow([epoch, avg_loss, avg_psnr])
        file.close()  
