'''
使用方法：修改以下变量
   example_path样本数据路径
   present_time
   time
   final_time
   j与time有关      3月加1    6月加2   9月加3
   stock_price,basic_financial_statement,dealed_financial_statement表格路径
   不得跨年
   '''
import openpyxl
import pandas as pd

# 给表格添加一行
# path="./data_set.xlsx"
# wb = openpyxl.load_workbook(path)
# ws = wb['Sheet1']
# ws.append(value)
# wb.save(path)

# 取表格特定位置的值
# excel_path1 = './1/table_first.xlsx'       #会读到Refreshing
# excel_path2 = './stock_price_2015.xlsx'
# print('k')
# d = pd.read_excel(excel_path1, sheetname=None,header=None)
# d=d['Sheet2']
# d=pd.DataFrame(d)
# print(d.shape)
# print(d.iloc[1,1])
# print(d.loc[16,0])
# print(d.loc[10,1])

#2015-03-31
present_time="2018-06-30"
time="2018-09-30"
final_time='2018-12-31'
example_path="./data_set/data_set_2018.xlsx"
wb1=openpyxl.load_workbook(example_path)
ws1=wb1['Sheet1']        #ws1用来添加样本数据集

stock_price_2015=pd.read_excel("./stock_price/stock_price_2018.xlsx",sheetname=None,header=None)
stock_price_2015=stock_price_2015['Sheet2']
stock_price_2015=pd.DataFrame(stock_price_2015)
#print(stock_price_2015)

basic_financial_statement=pd.read_excel("./financial_statement/basic_financial_statement_2018.xlsx",sheetname=None,header=None)
basic_financial_statement=basic_financial_statement['Sheet1']
basic_financial_statement=pd.DataFrame(basic_financial_statement)
#print(basic_financial_statement.iloc[9,1])

dealed_financial_statement=pd.read_excel("./financial_statement/dealed_financial_statement_2018.xlsx",sheetname=None,header=None)
dealed_financial_statement=dealed_financial_statement['Sheet2']
dealed_financial_statement=pd.DataFrame(dealed_financial_statement)
#print(dealed_financial_statement.iloc[9,1])

#获取100个样本数据
#每个样本数据的顺序为：1时间 2证券简称 3证券代码 4销售毛利率	 5销售净利率	6成本费用利用率 7净资产收益率	8总资产利润率	 9流动比率
# 10速动比率	11现金比率	12资产负债率	13产权比率	14已获利息倍数	15应收账款周转率	16存货周转率	17存货周转天数
# 18流动资产周转率	19固定资产周转率	20总资产周转率	21股东权益比率	22股东权益与固定资产比率	23每股净现金流量
# 24经营现金流与销售收入比率	25经营现金流与净利润比率	26每股净收益	27投资收益率	28每股净资产	29当前股价	30目标股价
# header=['时间','证券简称','证券代码','销售毛利率','销售净利率','成本费用利用率','净资产收益率','总资产利润率','流动比率',
#         '速动比率','现金比率','资产负债率','产权比率','已获利息倍数','应收账款周转率','存货周转率','存货周转天数','流动资产周转率',
#         '固定资产周转率','总资产周转率','股东权益比率','股东权益与固定资产比率','每股净现金流量','经营现金流与销售收入比率',
#         '经营现金流与净利润比率','每股净收益','投资收益率','每股净资产','当前股价','目标股价']

for i in range(100):
    value=[]
    j=i*5+3
    value.append(time)                                               #1时间
    value.append(stock_price_2015.iloc[2,i+1])                       #2证券简称
    value.append(stock_price_2015.iloc[1,i+1])                       #3证券代码
    value.append(dealed_financial_statement.iloc[53,j])              #4销售毛利率
    value.append(dealed_financial_statement.iloc[52,j])              #5销售净利率
    value.append(basic_financial_statement.iloc[53,j]/basic_financial_statement.iloc[16,j])  #6成本费用利用率
    value.append(dealed_financial_statement.iloc[39,j])              #7净资产收益率
    value.append(basic_financial_statement.iloc[58,j]/basic_financial_statement.iloc[222,j]) #8总资产利润率
    value.append(dealed_financial_statement.iloc[100,j])             #9流动比率
    value.append(dealed_financial_statement.iloc[102,j])             #10速动比率
    value.append(dealed_financial_statement.iloc[101,j])             #11现金比率
    value.append(dealed_financial_statement.iloc[88,j])              #12资产负债率
    value.append(dealed_financial_statement.iloc[104,j])             #13产权比率
    value.append(dealed_financial_statement.iloc[115,j])             #14已获利息倍数
    value.append(dealed_financial_statement.iloc[128,j])             #15应收账款周转率
    value.append(dealed_financial_statement.iloc[127,j])             #16存货周转率
    value.append(dealed_financial_statement.iloc[124,j])             #17存货周转天数
    value.append(dealed_financial_statement.iloc[130,j])             #18流动资产周转率
    value.append(dealed_financial_statement.iloc[131,j])             #19固定资产周转率
    value.append(dealed_financial_statement.iloc[132,j])             #20总资产周转率
    value.append(100-dealed_financial_statement.iloc[88,j])            #21股东权益比率
    value.append(basic_financial_statement.iloc[222,j]/basic_financial_statement.iloc[124,j]) #22股东权益与固定资产比率
    value.append(dealed_financial_statement.iloc[28,j])              #23每股净现金流量
    value.append(dealed_financial_statement.iloc[79,j])              #24经营现金流与销售收入比率
    value.append(basic_financial_statement.iloc[262,j]/basic_financial_statement.iloc[58,j])  #25经营现金流与净利润比率
    value.append(dealed_financial_statement.iloc[12,j])              #26每股净收益
    value.append(dealed_financial_statement.iloc[48,j])              #27投资收益率
    value.append(dealed_financial_statement.iloc[17,j])              #28每股净资产

    #29当前股价
    present_stock_price=0
    sum_stock_price=0
    count=0
    shape=stock_price_2015.shape
    for index in range(4,shape[0]):
        if str(stock_price_2015.iloc[index, 0])>present_time and str(stock_price_2015.iloc[index, 0])<=time:
            sum_stock_price+=stock_price_2015.iloc[index,i+1]
            count+=1.0
    present_stock_price=sum_stock_price/count
    value.append(present_stock_price)

    # 30目标股价
    target_stock_price = 0
    sum_stock_price = 0
    count = 0
    for index in range(4,shape[0]):
        if str(stock_price_2015.iloc[index, 0])>time and str(stock_price_2015.iloc[index, 0])<=final_time:
            sum_stock_price += stock_price_2015.iloc[index, i + 1]
            count += 1.0
    target_stock_price = sum_stock_price/count
    value.append(target_stock_price)
    #添加一个样本
    ws1.append(value)
wb1.save(example_path)