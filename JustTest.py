
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

plt.rcParams['font.sans-serif']=['SimHei']
def autolabel(rects,values):
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        plt.text(rect.get_x()+ rect.get_width()/2. - 0.05, height + 0.01, values[i])
def drawBarGraph(rank,name,wrongNumber,totalNumber,keys,values):
    valuesPercent = []
    for i in range(len(values)):
        valuesPercent.append(float(values[i])/float(totalNumber))
    bar = plt.bar(range(len(valuesPercent)), valuesPercent, color='steelblue', alpha=0.8)
    autolabel(bar,values)
    plt.xticks(rotation=20)
    plt.xticks(range(len(keys)), keys)
    plt.yticks(np.arange(0, 0.6, 0.1) )
    plt.title(name+" 识别错误比  "+str(wrongNumber)+"/"+str(totalNumber)+"  Rank:"+str(rank))
    plt.xlabel("类别名称")
    plt.ylabel("占比")
    plt.show()


content = [
[
    ["clean and jerk", "deadlifting", "squat", "snatch weight lifting", "driving tractor", "punching bag"],
    [41, 2, 2, 2, 1, 1],
    [8,49],
    ["clean and jerk"],
    [3],],
[
    ["javelin throw", "playing cricket", "high jump", "catching or throwing \nfrisbee", "throwing discus", "hammer throw"],
    [26, 3, 2, 2, 2, 1],
    [24,50],
    ["javelin throw"],
    [79],],
[
    ["hurdling", "high jump", "pole vault", "passing American \nfootball (in game)", "long jump", "motorcycling"],
    [25, 4, 3, 2, 2, 1],
    [25,50],
    ["hurdling"],
    [92],],
[
    ["swimming butterfly \nstroke", "swimming breast \nstroke", "swimming backstroke", "driving car", "jumping into pool", "snowkiting"],
    [21, 5, 4, 3, 2, 2],
    [29,50],
    ["swimming butterfly stroke"],
    [132],],
[
    ["triple jump", "long jump", "high jump", "hurdling", "jumping into \npool", "bouncing on \ntrampoline"],
    [17, 12, 4, 3, 1, 1],
    [32,49],
    ["triple jump"],
    [171],],
[
    ["driving car", "texting", "riding elephant", "driving tractor", "diving cliff", "milking cow"],
    [14, 3, 3, 3, 3, 2],
    [34,48],
    ["driving car"],
    [197],],
[
    ["playing harmonica", "playing recorder", "playing trombone", "eating spaghetti", "playing trumpet", "curling hair"],
    [14, 5, 2, 2, 2, 2],
    [35,49],
    ["playing harmonica"],
    [203],],
[
    ["hugging", "crawling baby", "smoking", "kissing", "petting animal \n(not cat)", "punching person \n(boxing)"],
    [12, 4, 3, 2, 2, 2],
    [37,49],
    ["hugging"],
    [223],],
[
    ["riding or walking \nwith horse", "riding mule", "riding elephant", "motorcycling", "crossing river", "riding camel"],
    [19, 10, 4, 3, 2, 2],
    [40,50],
    ["riding mule"],
    [262],],
[
    ["marching", "applauding", "giving or receiving\n award", "celebrating", "testifying", "singing"],
    [7, 6, 5, 5, 2, 2],
    [44,50],
    ["applauding"],
    [302],],
[
    ["dying hair", "braiding hair", "curling hair", "pumping fist", "brushing hair", "filling eyebrows"],
    [10, 5, 4, 3, 3, 1],
    [49,49],
    ["fixing hair"],
    [384],],
[
    ["scrambling eggs", "shuffling cards", "making pizza", "cooking chicken", "washing dishes", "doing nails"],
    [5, 4, 3, 2, 2, 2],
    [49,49],
    ["making a cake"],
    [385],],
]



for i in range(len(content)):
    item = content[i]
    drawBarGraph(item[4][0],item[3][0],item[2][0],item[2][1],item[0],item[1])