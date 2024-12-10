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
