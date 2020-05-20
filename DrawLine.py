
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import re

plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(20,10))

labels = ["add edge weight ","temporal weight","all weight"]
colors = ["r","g","b","darkilivegreen","teal"]
lossData = []
typeIndex = -1
iterMax = 300000

f = open("logData/AddEdgeWeight_2.txt")
line = f.readline()
while line:
    if line.find("Iter") != -1 :
        iter = int(re.compile(r'(?<=Iter )\d+\.?\d*').findall(line)[0])
        loss = re.compile(r'(?<=loss: )\d+\.?\d*').findall(line)[0]
        #print(iter,loss)
        if iter == 0:
            lossData.append([])
            lossData.append([])
            typeIndex = typeIndex + 1
        #print(typeIndex)
        #print(np.array(lossData).shape)
        lossData[2*typeIndex].append(float(iter))
        lossData[2*typeIndex+1].append(float(loss))
    line = f.readline()
f.close()

plt.yticks(np.arange(0, 6.3, 0.2) )
for i in range(int(len(lossData)/2)):
    tempMax = lossData[i * 2][-1]
    for j in range(len(lossData[i * 2])):
        lossData[i * 2][j] = int(lossData[i * 2][j] / tempMax * iterMax)


    #plt.plot(lossData[i*2],lossData[i*2+1],'o',label=labels[i]+"points")
    parameter = np.polyfit(lossData[i*2],lossData[i*2+1], 10)
    f = np.poly1d(parameter)  # 拼接方程
    plt.plot(lossData[i*2], f(lossData[i*2]),colors[i], label=labels[i])

plt.title('Loss的训练趋势')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.legend()
plt.show()
