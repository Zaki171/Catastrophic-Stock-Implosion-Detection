
import tsfel
import pandas as pd

def create_windows(df):
    cfg = tsfel.get_features_by_domain(json_path='features2.json') #DO NOT DO BOTH 
    result_dfs = []
    for fsym_id, group in df.groupby('fsym_id'):
        for i in range(10, len(group)+1):
            window = group.iloc[:i]
            X = tsfel.time_series_features_extractor(cfg, window.drop(['fsym_id', 'label'], axis=1), verbose=0)
            X['fsym_id'] = window['fsym_id'].iloc[0]
            X['label'] = window['label'].iloc[-1]
            X['end_date'] = window.index[-1]
            result_dfs.append(X)
            
    final_result = pd.concat(result_dfs, ignore_index=True)
    final_result.reset_index(drop=True, inplace=True)
    return final_result

        
        
df = pd.read_csv('df_b4_agg.csv')
new_df = create_windows(df) #might have to restrict implosions to stocks that have sufficient data!