import openpyxl
path='./accuracy_log.xlsx'
wb=openpyxl.load_workbook(path)
ws=wb['Sheet1']
value=['当前训练轮次','预测股价上涨','预测上涨且正确','预测股票下跌','预测下跌且正确','准确率']
ws.append(value)
value=[]
for i in range(6):
    value.append(i)
ws.append(value)
print(value)
wb.save(path)