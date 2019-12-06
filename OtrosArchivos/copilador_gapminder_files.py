import pandas as pd
import os

PATH='/home/david/Descargas/Gapminder data/'

files = os.listdir(PATH)

data = pd.DataFrame()
for item in files:
    if item.endswith('.csv'):
        print('Archivo:', item)
        temp = pd.read_csv(PATH+item)
        temp = temp.melt(id_vars='country', var_name="Date",
                value_name=item.split('csv')[0])
        if data.empty:
            data = temp
        else:
            data = pd.merge(data, temp, how='outer', on=['country', 'Date'])

data = data.sort_values(['country', 'Date'])

writer = pd.ExcelWriter(PATH+'copilado_gapminder.xlsx',  engine='xlsxwriter')
data.to_excel(writer,sheet_name = 'gap', index=False)
writer.save()
