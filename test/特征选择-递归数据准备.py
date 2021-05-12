import xlrd
import xlwt

datafileBase1 = '../sourseData/select_feature_data_1v4/'
datafileBase2 = '.xlsx'

# 打开文件
data = xlrd.open_workbook('../sourseData/31-60(newnocity1v4forone).xlsx')
# 获取表格
table = data.sheet_by_index(0)
# # 通过文件名获得工作表,获取工作表1
# table = data.sheet_by_name('Sheet1')

# 获取行数和列数
# 行数：table.nrows
# 列数：table.ncols
tableRows = table.nrows
tableCols = table.ncols
print("总行数：" + str(tableRows))
print("总列数：" + str(tableCols))

# 获取整行的值 和整列的值，返回的结果为数组
# 整行值：table.row_values(start,end)
# 整列值：table.col_values(start,end)
# 参数 start 为从第几个开始打印，
# end为打印到那个位置结束，默认为none
# print("整行值：" + str(table.row_values(0)))
# print("整列值：" + str(table.col_values(1)))

# 获取表头
# header = []
# headerIndex = 1
# while headerIndex < tableCols:
#     cel_Header = table.cell(0,headerIndex).value
#     headerIndex = headerIndex + 1
#     header.append(cel_Header)
# print('表头:' + str(header))

# 获取种子文献标号
# seedPaperIndex = []
# seedIndex = 1
# while seedIndex < tableRows:
#     cel_PaperIndex = table.cell(seedIndex,0).value
#     seedIndex = seedIndex + 1
#     seedPaperIndex.append(cel_PaperIndex)
# print('文献标号:' + str(seedPaperIndex))

# 获取某一种子的数据
seedPaperDetail = []
row = 1
rowWrite = 1
# 开始种子文献序号
init = 31
datafile = datafileBase1 + str(init) + datafileBase2;
print(datafile)
# 创建一个workbook 设置编码
workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('Sheet1')
# 保存
workbook.save(datafile)
# 获取表头
headerCol = 1
while headerCol < tableCols:
    cel_Header = table.cell(0,headerCol).value
    headerCol = headerCol + 1
    worksheet.write(0,headerCol-2 ,cel_Header)
    workbook.save(datafile)


while row < tableRows:
    col = 1
    colWrite = 0
    print(table.cell(row,0).value)
    seedIndex = table.cell(row,0).value
    workbook.save(datafile)

    if(init == seedIndex):
        print("==========")
        while col < tableCols:
            print(table.cell(row,col).value)
            worksheet.write(rowWrite,colWrite,table.cell(row,col).value)
            # worksheet.write(200,4,1232)
            col = col + 1
            colWrite = colWrite + 1
            workbook.save(datafile)
    else:
        rowWrite = 0
        init = init + 1
        datafile = datafileBase1 + str(init) + datafileBase2;
        print(datafile)
        # 创建一个workbook 设置编码
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet('Sheet1')
        # 保存
        workbook.save(datafile)
        # 获取表头
        headerCol = 1
        while headerCol < tableCols:
            cel_Header = table.cell(0,headerCol).value
            headerCol = headerCol + 1
            worksheet.write(0,headerCol - 2,cel_Header)
            workbook.save(datafile)
    row = row + 1
    rowWrite = rowWrite + 1



