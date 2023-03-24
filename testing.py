import torch
from model_scripts import data_saver

M = (28*28*64)
split = 0.05
N = int(0.005*M)

print("string length", N)
print("chunks =", M//N)

test_data = (torch.rand((N,))*255).type(torch.uint8)

print(test_data)

message, frequency_table = data_saver.arithmetic_encode(test_data)

n, d = message.as_integer_ratio()
print((n.bit_length()+7)//8)
print((d.bit_length()+7)//8)

print(frequency_table)
print(str(message)[:100])

decoded_data = data_saver.arithmetic_decode(frequency_table, message)

print(decoded_data)

correct = True

for i in range(N):
    if(test_data[i] != decoded_data[i]):
        correct = False

print("Success =",correct)
