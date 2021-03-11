import numpy as np
import pandas as pd
import csv


#Transforms the tsv to a csv format
def ssv_to_csv(filename):
    ur_outfile = 'vox1_meta.csv'
    with open(filename) as fin, open(ur_outfile, 'w') as fout:
        o=csv.writer(fout)
        for line in fin:
            o.writerow(line.split())

def main():
    df = pd.read_csv("vox1_meta.csv")
    #Only take the columns we need
    df = df[['ID', 'Nationality']]
    df.to_csv(r'vox1_nationality.csv')





if __name__ == "__main__":
    main()
