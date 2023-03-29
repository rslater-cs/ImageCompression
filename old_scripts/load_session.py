import os
import pandas as pd

def is_resume(session_path):
    if(os.path.exists(f'{session_path}/checkpoint.pt')):
        return True
    return False


if __name__ == "__main__":
    print(is_resume("./saved_models/SwinCompression_e12_t16_w2_d3/"))