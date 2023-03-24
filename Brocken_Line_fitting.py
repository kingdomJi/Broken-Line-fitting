import os.path

import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math


def cvtWhiteBleck(img):
    img[img == 255] = 1  # 黑白背景转换,0 是黑,255 是白
    img[img == 0] = 255
    img[img == 1] = 0
    return img

def cropHugeImg(img,size_h=300,size_w=300):
    img_np = np.asarray(img)  # np输入图像的格式(height(row*size_h),width(column*size_w))
    batchArray = []  # 存裁剪块的列表
    w = img_np.shape[1]  # 宽
    h = img_np.shape[0]  # 高
    #############计算行列数，有可能不被整除
    column = int(w / size_w)  # 块列数,按照去除边界的标准裁出的数量，即128*128能裁的数量算
    row = int(h  / size_h)  # 块行数
    for i in range(row):  # 先遍历行
        batchArray_inner = []  # 存每一行的结果
        for j in range(column):
            batch = img_np[i * size_h:(i + 1) * size_h,
                    j * size_w:(j + 1) * size_w ]
            # 截取img中某一块图像的格式是img[高1：高2，宽1：宽2]而不是img[高1：高2][宽1：宽2]
            # img_np[i]是遍历height,img_np[i][j]是遍历高和宽
            batchArray_inner.append(batch)
        batchArray.append(batchArray_inner)  # batchArray存最终裁剪结果
    batchArray = np.asarray(batchArray)
    return batchArray  # 返回的是一个数组size=(块行数,块列数,高，宽)
    #裁剪是从图像正上和正左方向开始的，待处理边角料在下方和右方以及右下角
def connectHugeImg(img_list,size_h=300,size_w=300):
    result=np.zeros(img_list.shape[0]*size_h,img_list.shape[1]*size_w)
    #结果图大小基于masks图片集大小,若有padding时加上边界sideLength*2
    for i in range(img_list.shape[0]):#遍历行
        for j in range(img_list.shape[1]):#遍历列
            result[i*(size_h):(i+1)*(size_h),j*(size_w):(j+1)*(size_w)]\
                =img_list[i,j]
    return np.asarray(result)#返回（classes,高，宽）的格式


def eightNeighborhoodDetection(img,inside):#对列表内点集范围内进行八邻域断点检测
    pointxy = list()
    # print(inside)
    for each in inside[0]:
        j = each[0]  # 宽
        i = each[1]#高,这里i,j对应要整清楚
        PX_PRE = img[i - 1]
        PX_curr = img[i]
        PX_Next = img[i + 1]
        p1 = PX_curr[j]
        if p1 != 255:
            continue
        p2 = PX_PRE[j]
        p3 = PX_PRE[j + 1]
        p4 = PX_curr[j + 1]
        p5 = PX_Next[j + 1]
        p6 = PX_Next[j]
        p7 = PX_Next[j - 1]
        p8 = PX_curr[j - 1]
        p9 = PX_PRE[j - 1]
        if p1 == 255:
            if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 255:
                pointxy.append([j, i])
    return pointxy#【输出宽，高】

def detection_CVline(img, x1,y1,x2,y2):#另一种找路径上已有点的尝试,x1,y1,x2,y2有问题？
    # print(img.shape)
    # print(x1,y1,x2,y2)#x1宽，y1高
    count=0
    img_t=np.zeros((img.shape[0],img.shape[1]))
    img_tt=np.zeros((img.shape[0],img.shape[1]))#test
    cv2.line(img_t,(x1,y1),(x2,y2),255,1)#选择宽为2就相当于画以3*3像素大小为点的线
    a = np.where(img_t == 255)[0].reshape(-1, 1)
    b = np.where(img_t == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([b, a], axis=1).tolist()  # 利用concatenate函数拼接，获取轮廓上及内部的点坐标
    inside = []  # 存储待检测区域的所有点
    inside.append(coordinate)
    # print(inside)
    for each in inside[0]:
        i = each[1]#
        j = each[0]
        img_tt[i][j]=255#test
        if(img[i][j]==255):
            count+=1
    print(count)
    return count


def eightNeighborhoodDetection_line(img, x1,y1,x2,y2):#对图像上给定两点间的连线以及连线邻域进行检测，返回路径上已有裂缝像素点个数
    count=0
    img_t = np.zeros((img.shape[0], img.shape[1]))#test
    for x in range(min(x1, x2)+1, max(x1, x2)):  # 中间路径，应该不含前也不含后
        y=int((x - x2) * (y1 - y2) / (x1 - x2) + y2)
        img_t[y-1][x-1]=255
        img_t[y ][x - 1] = 255
        img_t[y + 1][x - 1] = 255
        img_t[y - 1][x ] = 255
        img_t[y ][x ] = 255
        img_t[y + 1][x ] = 255
        img_t[y - 1][x + 1] = 255
        img_t[y ][x + 1] = 255
        img_t[y + 1][x + 1] = 255
        # img[y, x] = 255?
        PX_PRE = img[y - 1]
        PX_curr = img[y]
        PX_Next = img[y + 1]
        p1 = PX_curr[x]
        p2 = PX_PRE[x]
        p3 = PX_PRE[x + 1]
        p4 = PX_curr[x + 1]
        p5 = PX_Next[x + 1]
        p6 = PX_Next[x]
        p7 = PX_Next[x - 1]
        p8 = PX_curr[x - 1]
        p9 = PX_PRE[x - 1]
        if ((p1+p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 255):
            count+=int((p1+p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)/255)
    cv2.imshow('img',img_t)
    cv2.waitKey()
    return count
def findPoint(img,x1,y1,x2,y2):#根据给定的霍夫变换结果返回待拟合区域的四个角点，返回点集
    width=5#取线段周围多少像素范围
    """排除边界点的影响，不检测边界断点"""
    if(x1<=5 or x1>=img.shape[1]-5 or y1<=5 or y1>=img.shape[0]-5):#x对应shape[1],y对应shape[0]
        return 0
    elif(x2<=5 or x2>=img.shape[1]-5 or y2<=5 or y2>=img.shape[0]-5):
        return 0
        # pointxy = list()
        # img_size = img.shape  # height，width，passageway
    """ 计算斜率 """
    if (x2 != x1 and y1 != y2):  # 当不是垂线或横线
        k = (y2 - y1) / (x2 - x1)  # 原始线段斜率
        k2 = -1 / k  # 原始线段垂线的斜率
        dx = math.sqrt(1 / (k2 * k2 + 1)) * width  # dx肯定大于0，在原点x轴方向上的取值幅度（一半）,这里没有考虑dx<1的这种极端情况，这种属于特殊斜率情况
        dy = k2 * dx  # dy不确定正负
        # dx方+dy方=25
        # 四个点分别是(x1+dx,y1+dy)(x1-dx,y1-dy)(x2+dx,y2+dy)(x2-dx,y2-dy)
        point_1 = [x1 + dx, y1 + dy]
        point_2 = [x1 - dx, y1 - dy]
        point_3 = [x2 - dx, y2 - dy]
        point_4 = [x2 + dx, y2 + dy]
        ##当斜率特殊
    else:
        if (x2 == x1):  # and x1-width>=1 and x1+width<=img.shape[1]):#当原线段是垂线,含前不含后，x1-5要排除边界点
            point_1 = [x1 - 5, y1]
            point_2 = [x1 + 5, y1]
            point_3 = [x2 + 5, y2]
            point_4 = [x2 - 5, y2]
        elif (y1 == y2):  # and y1-width>=1 and y1+width<=img.shape[0]):#当是横线，且排除边界点
            point_1 = [x1, y1 + 5]
            point_2 = [x2, y1 + 5]
            point_3 = [x2, y1 - 5]
            point_4 = [x1, y1 - 5]
    # print(np.array([point_1, point_2, point_3, point_4]).astype(int))  # 是输出4个点
    contours = np.array([point_1, point_2, point_3, point_4]).astype(int)  # 元组，返回四个角点,注意x与y的顺序
    # print(contours)
    img_t = np.zeros((img.shape[0], img.shape[1]), np.uint8)#临时变量
    cv2.drawContours(img_t, [contours], -1, 255, thickness=-1)  # 画四边形
    a = np.where(img_t == 255)[0].reshape(-1, 1)
    b = np.where(img_t == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([b, a], axis=1).tolist()  # 利用concatenate函数拼接，获取轮廓上及内部的点坐标
    inside = []  # 存储待检测区域的所有点
    inside.append(coordinate)

    return inside#返回待检测区域的所有点


def connectPoint(img,Pointlist,x,y):
    x0=x
    y0=y
    dict={}
    for each in Pointlist:#计算距离
        dict['{0},{1}'.format(each[0],each[1])]=math.pow(each[1]-y0,2)+math.pow(each[0]-x0,2)
        # length=math.pow(each[1]-y0,2)+math.pow(each[0]-x0,2)#不确定对不对
    tuplelist = list(zip(dict.values(),dict.keys()))
    #使用sorted函数对元组列表list由距离低到高进行排序
    tuplelist_sorted = sorted(tuplelist)#排完序的点与原点距离
    # print(tuplelist_sorted)
    dict_sorted = {}
    for rank, (key, point_val) in enumerate(tuplelist_sorted, 1):#设置下标从1开始
        # 重新构造带有排名的排序后的字典dict_sorted
        point_val=[int(point_val.split(',')[0]),int(point_val.split(',')[1])]
        # print(point_val)#[宽，高]
        dict_sorted['{}'.format(rank)] = point_val

    i=False#控制路径临时参数
    for val in dict_sorted.values():#遍历排好序的点坐标字典
        if(i==False):
            t1=val[0]#宽
            t2=val[1]#高
            i=True
        else:
       
            count=detection_CVline(img,t1,t2,val[0],val[1])
            # print(count)
            if(count>3):#设置适合的阈值，超过阈值则不连接
                t1=val[0]#后点变前点
                t2=val[1]
                continue
            cv2.line(img,(t1,t2),(val[0],val[1]),255,1)#划线
            i=False

def HoughLinesP_Jing(img,contrast_img_path,rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None,contrast=True):
    if contrast==True:
        img_contrast = cv2.imread(r'{}'.format(contrast_img_path), 1)  # 效果对比图之断点，1是以BGR图方式去读
        img_contrast = cvtWhiteBleck(cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB))  # 转RGB+黑白转换
        img_contrast2 = cvtWhiteBleck(cv2.imread(r'{}'.format(contrast_img_path), 0))  # 效果对比图之断点检测域

    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    for line in lines:  # 读取所有线，每次是处理一条预测线路的情况
        x1, y1, x2, y2 = line[0]  # 读取端点坐标
        # 返回待检测区域内部点集
        inside = findPoint(img_test, x1, y1, x2, y2)  # 输出【宽，高】
        # print(inside)  # 输出【宽，高】
        if inside == 0:  # 排除边界点
            continue
        #################计算对比图
        for each in inside[0]:
            j = each[0]  # 宽
            i = each[1]  # 高,这里i,j对应要整清楚
            img_contrast2[i][j] = 255
        ####################
        # 八邻域断点检测
        pointxy = eightNeighborhoodDetection(img_test, inside)  # 输出pointxy是断点集【宽，高】
        if len(pointxy) > 1:  # 对检测出的断点进行连接,排除没有或单个断点的情况
            connectPoint(img_test, pointxy, x1, y1)  # 连接断点，宽横向x1=shape[1]，高纵向y1=shape[0]
            for each in pointxy:  # 在对比图上画被标记的断点
                img_contrast[each[1], each[0]] = [0, 0, 255]
    img_reslut = img_test
    if contrast == True:
        return img_reslut,img_contrast,img_contrast2
    else:
        return img_reslut


if __name__=='__main__':
    dir_path=r'C:\Users\Administrator\Desktop\U-net\Pytorch-UNet\utils\HoughLinesJ_test'#处理该文件夹下的图片
    result_path=r'C:\Users\Administrator\Desktop\U-net\Pytorch-UNet\utils\HoughLinesJ_result'#输出结果到该文件夹
    dirs=os.listdir(dir_path)
    for each in dirs:#each是单张图片名，遍历文件夹下的图片们
        img_path=dir_path+'\\{0}'.format(each)
        print(img_path)
        img_test = cv2.imread(img_path, -1)
        img_test=cvtWhiteBleck(img_test)#底图的黑白像素转换
        if img_test.shape[0]>1000 and img_test[1]>1000:#针对特大图的情况，未完善
            img_list=cropHugeImg(img_test)#size=(块行数,块列数,高，宽)
            for i in img_list:#遍历行
                for img_batch in i:#遍历列
                    lines =HoughLinesP_Jing(img_batch, 1, np.pi / 180, 10, minLineLength=20, maxLineGap=20)#概率霍夫变换
                        # for each in pointxy:  # 在对比图上画被标记的断点
                        #     img_contrast[each[1], each[0]] = [0, 0, 255]
            img_reslut=connectHugeImg(img_list)
        else:
            img_reslut,img_contrast,img_contrast2 = HoughLinesP_Jing(img_test,img_path, 1, np.pi / 180, 10, minLineLength=20, maxLineGap=20,contrast=True)
            img_reslut2, img_contrast, img_contrast2 = HoughLinesP_Jing(img_reslut, img_path, 1, np.pi / 180, 20,#二次霍夫与拟合变换
                                                                       minLineLength=40, maxLineGap=40, contrast=True)
        # img_reslut3, img_contrast, img_contrast2 = HoughLinesP_Jing(img_reslut2, img_path, 1, np.pi / 180, 30,# 三次霍夫与拟合变换
        #                                                             minLineLength=60, maxLineGap=60, contrast=True)
        retval = cv2.imwrite(result_path+'\\{0}'.format(each)+'_AfterConnect.png',img_reslut2)
  





