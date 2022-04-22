import xlwt

# 创建一个可以写入每层计算时间的excel表
workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('My Worksheet')
array_watch = [0, 0.006062644, 0.021762818, 0.024211551, 0.041808019, 0.044681877, 0.045140355, 0.049571071, 0.056326208,
         0.063487921, 0.063761975, 0.064497671, 0.064867865, 0.065033272]

array_phone = [0.006478532, 0.006135402, 0.004713426, 0.004367099, .002545778, 0.002308766, 0.002244584, 0.001707029,
               0.000953468, 0.000182891, 0.000145245, 6.86169E-05, 3.05732E-05, 0]

trasmition_time = [0.107086182, 0.219726563, 0.219726563, 0.047851563, 0.03515625, 0.03515625, 0.006103516, 0.012207031,
                   0.012207031, 0.012207031, 0.001953125, 0.001464844, 0.000732422, 3.8147E-05]
temptramition_time = [0 for i in range(15)]
endedge_time = [0 for i in range(50)]
end_time = [0.065033272 for i in range(50)]
edge_time = [0 for i in range(50)]
data = 0.107086182  # 字节数

for i in range(1, 50):
    bandwidth = i/8  # 如果这里是mbps
    min = float("inf")
    for j in range(0, 14):
        temptramition_time[j] = trasmition_time[j]/bandwidth
        if temptramition_time[j]+ array_watch[j] + array_phone[j] < min:
              min = temptramition_time[j] + array_watch[j] + array_phone[j]
    endedge_time[i] = min
    edge_time[i] = array_phone[0] + data/bandwidth
    worksheet.write(0, i, endedge_time[i])  # 将时间写入列表
    worksheet.write(1, i, end_time[i])
    worksheet.write(2, i, edge_time[i])
    workbook.save('inference_latency.xls')  # 保存文件

