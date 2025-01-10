import numpy as np
import services.process_pivot_points as pp


def process_data(df):
    # pe_mean = df['市盈率'].mean()
    # pe_median = df["市盈率"].median()
    # pb_mean = df["市净率"].mean()
    # pb_median = df["市净率"].median()
    # df["市盈率分位点"] = df["市盈率"].rank(pct=True)
    # df["市净率分位点"] = df["市净率"].rank(pct=True)    

    pivot_points = df.apply(lambda row: pp.calculate_pivot_points(row['最高价'], row['最低价'], row['收盘价']), axis=1)
    
    df["枢轴点"] = pivot_points.apply(lambda x: x["Pivot Point"])
    df["第一阻力位"] = pivot_points.apply(lambda x: x["Resistance 1"])
    df["第一支撑位"] = pivot_points.apply(lambda x: x["Support 1"])
    df["第二阻力位"] = pivot_points.apply(lambda x: x["Resistance 2"])
    df["第二支撑位"] = pivot_points.apply(lambda x: x["Support 2"])
    df["第三阻力位"] = pivot_points.apply(lambda x: x["Resistance 3"])
    df["第三支撑位"] = pivot_points.apply(lambda x: x["Support 3"])

    return df


