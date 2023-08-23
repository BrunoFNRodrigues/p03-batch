import pandas as pd
from datetime import datetime

db = pd.read_csv("../data/train-2023-08-01.csv")
new_db = db.copy()
new_db = new_db.drop(["date"], axis=1)

new_db[['year', 'month', 'day']] = db.date.str.split("-", expand = True)

date_format = '%Y-%m-%d'
new_db[['weekday']] = db['date'].apply(lambda x: pd.Series(datetime.strptime(str(x), date_format).weekday()))

new_db = new_db.groupby(['store_id','day', 'month', 'year', 'weekday']).price.sum().reset_index()
new_db.rename(columns={'price':'total_sales'}, inplace=True)
print(new_db)
new_db.to_parquet("../data/train-2023-08-01.parquet",index=False)
