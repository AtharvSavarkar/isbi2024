
import json
import pathlib
import requests
from utils import custom_random_dataset, train_all_LF

def get_variables():
    
    # f = open("/home/atharvs/PycharmProjects/isbi2024/code/config.json")
    f = open("../code/config.json")
    data = json.load(f)   
    classes = data["classes"]
    label_frac = data["label_frac"]
    path = data["path"]
    save_path = data["save_path"]
    return classes, label_frac, path, save_path


if __name__ == "__main__":
    
    classes, label_frac, path, save_path = get_variables()
    print(classes)
    num_cls = len(classes)
    print(num_cls)
    dataset, x, y = custom_random_dataset(classes=classes, path=path, fraction=label_frac)
    train_all_LF(x, y, num_cls, path=save_path+str(int(label_frac*100)), fraction=label_frac)
