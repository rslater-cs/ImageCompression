class Printer:
    def __init__(self, path, name="output", mode='w') -> None:
        self.path = f'{path}/{name}.txt'
        file = open(self.path, mode)
        file.close()

    def print(self, message):
        file = open(self.path, 'a')
        file.write(f'{message}\n')
        file.close()

class Status(Printer):
    def __init__(self, path) -> None:
        super().__init__(path, name="status")
        
    def print(self, message):
        file = open(self.path, 'w')
        file.write(f'{message}\n')
        file.close()

