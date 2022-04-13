import function


if __name__ == '__main__':
    path = "../res/test.xls"
    sheet = "sheet1"
    value = [["传输时间2","传输大小2"]]
    #
    function.create_excel_xsl(path,sheet,value)
    function.write_excel_xls_append(path,sheet,value)
    # function.read_excel_xls(path,sheet)


