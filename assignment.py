import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def remove_text_based_features(df_src:pd.DataFrame) -> pd.DataFrame:
    """
    Returns:
    Dataframe that only contains coulmns of type float.
    """
    df_dst = df_src.select_dtypes(include = ["float"])
    return df_dst

def knn_impute_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Returns:
    A copy of (df) with missing values that have been imputed using a nearest neighbor algorithm 
    """
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    columns = df.columns
    index = df.index
    return pd.DataFrame(imputer.fit_transform(df), columns=columns, index=index)

def comparison_score(arr1:list, arr2:list) -> float:
    """
    Calculate the R-squared value between two arrays.

    Returns:
    float: R-squared value.
    """
    u = ((arr1 - arr2) ** 2).sum()
    v = ((arr1 - arr1.mean()) ** 2).sum()
    score = 1 - u / v
    return score

def convert_to_one_hot_encoding(df_src:pd.DataFrame) -> pd.DataFrame:
    # Enter code for challenge below
    raise NotImplementedError

def normalize_data(df_src:pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    # Enter code for challenge below
    raise NotImplementedError

def rescale_data(df_src:pd.DataFrame, scaler:MinMaxScaler):
    # Enter code for challenge below
    raise NotImplementedError

def make_regression_model(df:pd.DataFrame) -> RandomForestRegressor:
    # Enter code for challenge below
    raise NotImplementedError

def model_missing_data(model:RandomForestRegressor, df:pd.DataFrame) -> pd.DataFrame:
    # Enter code for challenge below
    raise NotImplementedError

def populate_missing_data(df_src:pd.DataFrame, one_hot:bool=False, normalize:bool=False, population_method:str="knn") -> pd.DataFrame:
    """
    Populates missing 'price' data in a the given dataframe via a selection of methods.

    Args:
        df_src (Dataframe): Given dataframe.
        one_hot (bool): Flag to convert text data into numerical data using one-hot encoding. 
                [False] removes columns with non-numerical data. [True] replaces those columns with one-hot encoded columns
        normalize (bool): Flag to normalize numerical data for imputation.
        impute_method (str): How to populate missing price data.
                ['knn'] uses sklearn.impute.KNNImputer ['regression'] uses sklearn.ensemble.RandomForestRegressor
    Returns:
        A copy of the given dataframe where the missing price values have been populated
    """
    df_dst = df_src.copy()
    if one_hot:
        df_dst = convert_to_one_hot_encoding(df_dst)
    else:
        df_dst = remove_text_based_features(df_dst)

    if normalize:
        df_dst,scaler = normalize_data(df_dst)

    match population_method:
        case "knn":
            df_dst = knn_impute_missing_data(df_dst)
        case "regression":
            model = make_regression_model(df_dst)
            df_dst = model_missing_data(model, df_dst)
        case _:
            raise ValueError(f"{population_method} is not a recognized population method")
        
    if normalize:
        df_dst = rescale_data(df_dst, scaler)

    return df_dst

if __name__ == '__main__':    
    df_in = pd.read_csv("./data/raw_diamond_data.csv", index_col=0)
    # Get indicies of data entries with missing price information
    missing_price_index = df_in[df_in["price"].isnull()].index.tolist()

    # Validate comparison_score function works as expected by testing the identity 
    score = comparison_score(df_in.loc[df_in.index.difference(missing_price_index),"price"],
                              df_in.loc[df_in.index.difference(missing_price_index),"price"])
    print(f"Identity comparison score is {score:.3f}.")

    # Run the missing data population algorithm
    df_out = populate_missing_data(df_in, one_hot=False, normalize=False, population_method="knn")

    # Get the full data score
    full_df = pd.read_csv("./data/complete_diamond_data.csv", index_col=0)
    score = comparison_score(full_df.loc[missing_price_index,"price"],
                              df_out.loc[missing_price_index,"price"])
    print(f"R-squared value for populating missing prices is {score:.3f}")
    
