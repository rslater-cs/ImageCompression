# Class for easy printing to a log file, used to track 
# AI@Surrey server progress
class Printer:
    def __init__(self, path, name="output", mode='w') -> None:
        self.path = f'{path}/{name}.txt'
        file = open(self.path, mode)
        file.close()

    def print(self, message):
        file = open(self.path, 'a')
        file.write(f'{message}\n')
        file.close()

# Class for printing only the last line to a log file
# Overwrites every time a message is added
class Status(Printer):
    def __init__(self, path) -> None:
        super().__init__(path, name="status")
        
    def print(self, message):
        file = open(self.path, 'w')
        file.write(f'{message}\n')
        file.close()

