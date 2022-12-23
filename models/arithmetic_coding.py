import torch

class AC():

    def encode(self, data: torch.Tensor):
        data = data.flatten()
        freq_map = self.create_map(data)
        probs = self.create_probs(freq_map)


    def create_map(self, data: torch.Tensor):
        val_map = dict()

        for item in data:
            if item in val_map:
                val_map[item] += 1
            else:
                val_map[item] = 1

        return val_map

    def create_probs(self, freqs):
        probs = torch.empty(len(freqs))

        total = torch.sum(torch.tensor(freqs.values()))

        i = 0
        for key, value in freqs:
            probs[i] = value/total
            i+=1

        return probs

    def calc_interval(probs: torch.Tensor, prob_index: int):
        S = torch.sum(probs[0:prob_index])
        R = torch.sum(probs)
        
