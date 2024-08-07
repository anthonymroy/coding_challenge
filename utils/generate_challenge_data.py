import pandas as pd
import random

FULL_DATA_FILEPATH = "./data/diamonds_complete.csv"
CHALLENGE_DATA_FILEPATH = "./data/diamonds_incomplete.csv"

def remove_random_values(df_src:pd.DataFrame, column:str, removal_rate:float) -> pd.DataFrame:
    entries = list(range(df_src.shape[0]))    
    random.shuffle(entries)
    entries = entries[0:int(removal_rate*len(entries))]
    df_dst = df_src.copy()
    for i in entries:
        df_dst.at[i,column] = pd.NA
    return df_dst

if __name__ == '__main__':
    df = pd.read_csv(FULL_DATA_FILEPATH,index_col=0)
    df = remove_random_values(df, "price", 0.2)
    df.to_csv(CHALLENGE_DATA_FILEPATH)