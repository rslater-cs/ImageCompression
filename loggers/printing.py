class Printer:
    def __init__(self, path, id, name="output") -> None:
        self.path = f'{path}/{name}_{id}.txt'
        file = open(self.path, 'w')
        file.close()

    def print(self, message):
        file = open(self.path, 'a')
        file.write(f'{message}\n')
        file.close()

class Status(Printer):
    def __init__(self, path, id) -> None:
        super().__init__(path, id, name="status")
        
    def print(self, message):
        file = open(self.path, 'w')
        file.write(f'{message}\n')
        file.close()

