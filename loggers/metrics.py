import csv

class MetricLogger:
    
    def __init__(self, path, name, size, mode='w'):
        path = f'{path}/{name}.csv'
        self.file = open(path, mode, newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch", f"{name}_loss", f"{name}_psnr"])
        self.size = size

    def put(self, epoch, total_loss, total_psnr):
        avg_loss = total_loss/self.size
        avg_psnr = total_psnr/self.size

        self.writer.writerow([epoch, avg_loss, avg_psnr])

    def close(self):
        self.file.close()   
