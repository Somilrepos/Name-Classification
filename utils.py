import io 
import os
import subprocess
import torch
import pandas as pd
import time

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',"
N_LETTERS = len(LETTERS)
labels = []

# Thanks to GKG https://www.geeksforgeeks.org/convert-unicode-to-ascii-in-python/
def unicode_to_ascii(s):
    process = subprocess.Popen(['iconv', '-f', 'utf-8', '-t', 'ascii//TRANSLIT'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate(input=s.encode())
    if not error:
        ascii_string = output.decode()
        return ascii_string
    else:
        raise Exception(error)

def loadata():
    """ Helper funtion for loading data
    inputs: 
        NONE
    
    outputs:
    @param category_lines(list): list of names.
    @param category_labels(list): list of name categoriee for respective name in @category_lines
    @param categories(list): list of unique name categories.
    """
    category_lines  = []
    category_labels = []
    categories = []

    print("DataLoading Started..")
    st = time.time()

    for (root,_,files) in os.walk(os.getcwd()+"/data/names"):

        for file in files:
            category_name = file.split(".")[0]
            filepath = root+"/" + file
            
            lines = io.open(filepath, encoding="utf-8").read().split("\n")

            category_lines.extend([unicode_to_ascii(line.lower()) for line in lines])
            category_labels.extend([category_name]*len(lines)) 

            categories.append(category_name)
        et = time.time()

    df = pd.DataFrame(list(zip(category_lines, category_labels)))
    df = df.loc[df[0] != '']
    
    print(f"DataLoading Ended... Total Execution Time:{(et-st)} seconds")
    return df[:][0], df[:][1], categories

def line_to_tensor(line):
    """ Converts a line into a tensor

    inputs:
    @param line(stirng): line contaning name 
    
    output:
    @param line_tensor(torch.tensor): <line.length() * 1 * N_LETTERS> One hot representation of every charachter in a line

    """
    line_tensor = torch.zeros(len(line),1,N_LETTERS)

    for i, letter in enumerate(line):
        line_tensor[i][0][LETTERS.find(letter)] = 1
    
    return line_tensor

def label_to_tensor(label, labels):
    tensor = torch.zeros(1,len(labels))
    tensor[labels.index(label)] = 1
    return tensor

if __name__== "__main__":
    print("Welcome to the AI/ML world!")