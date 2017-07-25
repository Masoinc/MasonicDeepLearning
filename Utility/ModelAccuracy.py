def getsumse(predict, real):
    SumSE = 0
    for i in range(len(real)):
        SE = pow(predict[i] - real[i], 2)
        SumSE += SE
    return SumSE


def getrsqured(predict, real):
    sum = 0
    sumdevsqr = 0
    for i in real:
        sum += i
    mean = sum / len(real)
    for i in real:
        sumdevsqr += pow((i - mean), 2)
    Rsqured = 1 - (getsumse(predict, real) / sumdevsqr)
    return Rsqured