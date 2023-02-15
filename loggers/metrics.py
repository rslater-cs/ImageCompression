import csv

class MetricLogger:
    
    def __init__(self, path, name, size, mode='w'):
        self.path = f'{path}/{name}.csv'
        file = open(self.path, mode, newline='')
        writer = csv.writer(file)
        if(mode=='w'):
            writer.writerow(["epoch", f"{name}_loss", f"{name}_psnr"])
        self.size = size

    def put(self, epoch, total_loss, total_psnr):
        file = open(self.path, 'a', newline='')
        writer = csv.writer(file)
        avg_loss = total_loss/self.size
        avg_psnr = total_psnr/self.size

        writer.writerow([epoch, avg_loss, avg_psnr])
        file.close()  
