import tsfel
import pandas as pd

def feature_extraction(df):
    df = df.set_index('date')
    exclude_columns = ['label']
    
    X_train = df.drop(exclude_columns, axis=1)
    
    cfg = tsfel.get_features_by_domain("statistical")
    
    result_dfs = []
    for fsym_id, group_df in X_train.groupby('fsym_id'):
        # Exclude 'fsym_id' column from group_df
        # print(group_df.head())
        non_zero_cols = group_df.columns[(group_df != 0).any()]
        group_df = group_df[non_zero_cols]

        if not group_df.empty:
            try:
                X = tsfel.time_series_features_extractor(cfg, group_df.drop('fsym_id', axis=1), verbose=0);
                result_dfs.append(X)
            except ValueError:
                continue
    
    final_result = pd.concat(result_dfs, ignore_index=True)
    final_result.reset_index(drop=True, inplace=True)
    return final_result

def more_feats(df):
    df1 = feature_extraction(df)
    return df1

