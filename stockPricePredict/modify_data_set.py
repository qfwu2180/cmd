import pandas as pd
data2015=pd.read_excel("./data_set/data_set_2015.xlsx")
data2016=pd.read_excel("./data_set/data_set_2016.xlsx")
data2017=pd.read_excel("./data_set/data_set_2017.xlsx")
data2018=pd.read_excel("./data_set/data_set_2018.xlsx")

# for i in range(data2015.shape[0]):
#     data2015.ix[i,'股东权益比率']=100-data2015.ix[i,'资产负债率']
#
# pf=pd.DataFrame(data2015)
# pf.to_excel('./aa.xlsx')

for i in range(data2016.shape[0]):
    data2016.ix[i,'股东权益比率']=100-data2016.ix[i,'资产负债率']
pf=pd.DataFrame(data2016)
pf.to_excel('./bb.xlsx')

for i in range(data2017.shape[0]):
    data2017.ix[i,'股东权益比率']=100-data2017.ix[i,'资产负债率']
pf=pd.DataFrame(data2017)
pf.to_excel('./cc.xlsx')

for i in range(data2018.shape[0]):
    data2018.ix[i,'股东权益比率']=100-data2018.ix[i,'资产负债率']
pf=pd.DataFrame(data2018)
pf.to_excel('./dd.xlsx')