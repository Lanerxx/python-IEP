wGap = 0.01 # 权重增幅（越少越精确，耗时越长）x
wSum = 0
wCount = 0
indexFeature = 0
featureWeighted = [0.5,0.2]
while indexFeature < len(featureWeighted):
    wSum += featureWeighted[indexFeature] + 0.5
    indexFeature += 1
print(wSum / 0.01)

# 在不筛选特征的情况下的初始f1
indexFeature = 0
while indexFeature < len(featureWeighted):
    print("============")
    w = 0
    while w < featureWeighted[indexFeature] + 0.5:
        w += wGap
        wCount += 1
        print(wSum)
        print(wCount)
        print("当前进度：" + str(wCount / wSum))
        print("%.2f%%" % (wCount / wSum ))
    indexFeature += 1
