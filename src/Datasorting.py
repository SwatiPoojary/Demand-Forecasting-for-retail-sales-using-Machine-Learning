import  pandas as pd

def month_number (row):
   if row['Month_Name'].strip() == 'Jan' :
        return 1
   elif row['Month_Name'].strip() == 'Feb':
        return 2
   elif row['Month_Name'].strip() == 'Mar':
        return 3
   elif row['Month_Name'].strip() == 'Apr':
        return 4
   elif row['Month_Name'].strip() == 'May':
        return 5
   elif row['Month_Name'].strip() == 'Jun':
        return 6
   elif row['Month_Name'].strip() == 'Jul':
        return 7
   elif row['Month_Name'].strip() == 'Aug':
        return 8
   elif row['Month_Name'].strip() == 'Sep':
        return 9
   elif row['Month_Name'].strip() == 'Oct':
        return 10
   elif row['Month_Name'].strip() == 'Nov':
        return 11
   elif row['Month_Name'].strip() == 'Dec':
        return 12
   return '0'

xl_file = pd.ExcelFile('..\data\poundsdata.xlsx');

dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
data = dfs['ValNSAT']
data = data.iloc[9:,[0,8]]
data.columns = ['Date','Cost']
data[['Year','Month_Name']] = data['Date'].str.split(' ', 1, expand=True)
data['Month'] = data.apply (lambda row: month_number(row), axis=1)
data['Day'] = 1
cols=["Day","Month","Year"]
data['OrderDate'] = data[cols].apply(lambda x: '/'.join(x.values.astype(str)), axis="columns")
print(data)
data.to_csv('..\data\ProcessedData.csv', index = False)
# print(pd.to_datetime((data.Year*10000+data.Month*100+1).apply(str),format='%Y%m%d'))
