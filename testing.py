import os

root = ".\\saved_models\\"
networks = [(os.path.join(root, name), int(name.split("_")[1])) for name in os.listdir(".\\saved_models\\") if os.path.isdir(os.path.join(root, name))]

networks.sort(key= lambda x: x[1])

max_id = networks[-1][1]

network_id = max_id+1