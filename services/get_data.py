import akshare as ak
import pandas as pd


def get_shenwan_data():
	big_df = ak.index_analysis_daily_sw(symbol="二级行业", start_date="20240101", end_date="20240103")

	# input data into a file as xls

	big_df.rename(
		columns={
			"swindexcode": "指数代码",
			"swindexname": "指数名称",
			"bargaindate": "发布日期",
			"pe": "市盈率",
			"pb": "市净率",
		},
		inplace=True
	)

	big_df["发布日期"] = pd.to_datetime(big_df["发布日期"]).dt.strftime("%Y-%m-%d")
	filtered_df = big_df[["指数代码", "指数名称", "发布日期", "市盈率", "市净率"]]

	return filtered_df

def get_hs_data(start, end):
	big_df = ak.stock_hk_index_daily_sina(symbol="HSI")
	big_df.rename(
        columns={
            "high": "最高价",
            "low": "最低价",
            "open": "开盘价",
            "close": "收盘价",
			"date": "日期",
        },
        inplace=True
    )
	big_df['日期'] = pd.to_datetime(big_df['日期'])
    
    # 过滤特定时间范围的数据
	filtered_df = big_df[(big_df['日期'] >= start) & (big_df['日期'] <= end)]
    
    # 选择需要的列
	filtered_df = filtered_df[["日期", "最高价", "最低价", "开盘价", "收盘价"]]

	return filtered_df