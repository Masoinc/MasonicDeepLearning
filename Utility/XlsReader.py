import xlrd as xlrd
import numpy as np


def readxlsbycol(directory, sheet, col):
    xls = xlrd.open_workbook(directory)
    s = xls.sheet_by_name(sheet)
    cols = s.col_values(col)

    return cols


def readxlsbyrow(directory, sheet, row):
    xls = xlrd.open_workbook(directory)
    s = xls.sheet_by_name(sheet)
    rows = s.row_values(row)
    return rows


def readxls(directory, sheet):
    xls = xlrd.open_workbook(directory)
    s = xls.sheet_by_name(sheet)
    return s


def getdivideddataset(sheet, tscale):
    x, xtest, y, ytest = [], [], [], []

    for row in range(sheet.nrows):
        if np.random.random(1) > tscale:
            xtest.append(sheet.row(row)[0].value)
            ytest.append(sheet.row(row)[1].value)
        else:
            x.append(sheet.row(row)[0].value)
            y.append(sheet.row(row)[1].value)
    return x, xtest, y, ytest


if __name__ == '__main__':
    DIRECTORY = r"E:\PyCharmProjects\MasonicDeepLearning\DataSet\Bailuyuan.xlsx"
    SHEET = "Sheet1"

    xtrain, xtest, ytrain, ytest = getdivideddataset(readxls(DIRECTORY, SHEET), 0.7)
    print(xtrain)
