from torch.utils.tensorboard.plugins.hparams import api as hp

import yaml
def read_file(filename):
    with open(filename,'r') as f:
        data = list(yaml.safe_load_all(f))
        return data
data = read_file('./configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml')[0]

def write_file(py_obj, filename):
    with open(filename, 'w') as f :
        yaml.dump(py_obj, f, sort_keys=False) 
    print('Written to file successfully')
write_file(data, './configs/rec/PP-OCRv3/test.yml')