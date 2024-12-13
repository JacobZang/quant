def process_data(df):
    # pe_mean = df['市盈率'].mean()
    # pe_median = df["市盈率"].median()
    # pb_mean = df["市净率"].mean()
    # pb_median = df["市净率"].median()

    df["市盈率分位点"] = df["市盈率"].rank(pct=True)
    df["市净率分位点"] = df["市净率"].rank(pct=True)    

    return df