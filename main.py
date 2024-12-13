import services.get_data as get_data
import services.process_data as process_data

df = get_data.get_shenwan_data()
df = process_data.process_data(df)
