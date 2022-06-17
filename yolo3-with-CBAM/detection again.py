import cv2
import time
import numpy as np
def detection_again(img,img_num):
    obj = 0
    if not img_num:#帧数
        previous = img

    gray_diff = cv2.absdiff(img, previous)#计算两帧之间的像素差
    thresh = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)[1]
    point_change = (np.array(thresh) > 0).sum()#转换为数组,求得变换像素点个数
    if point_change > 20:#如果变化像素点大于20 判断为存在运动物体
        obj = 1
    return obj
img = cv2.imread('E:/dataset/(115).jpg')
#模板匹配 多目标匹配 一个模板去匹配图片上相似的多个目标 返回目标坐标，多模板匹配： 多个模板去匹配图片上的目标，看是否存在
result = cv2.matchTemplate()
#滤波 滤波器对每个像素进行操作
#1.均值滤波 噪声像素面积小，用均值的方式，把滤波器（3*3)内的像素值取平均，能够去噪，
cv2.blur()
#2.中值滤波 滤波器内部像素按序排列，取中间部分给核心像素
cv2.medianBlur()
#3.高斯滤波 离滤波器核心越近 权重越大
cv2.GaussianBlur()
#4.双边滤波 去噪同时 保留边缘信息
cv2.bilateralFilter()
#开运算：腐蚀，膨胀，抹除图像外部噪点  闭运算：膨胀，腐蚀，抹除图像内部噪点
#梯度运算 ：膨胀图减去腐蚀图 得到大概轮廓
#顶帽运算：原图减去原图 的开运算图，得到外部噪点 or细节
#黑帽3运算： 原图的闭运算图减去原图 ，得到内部细节


#图形检测 需提前降噪
contours, hierarchy = cv2.findContours(img, mode, methode)#返回轮廓list格式，和轮廓层次信息
img = cv2.drawContours(image, contours,)#轮廓画个框
retval = cv2.boudingRect(array)#计算轮廓最小矩形边界 返回左上角顶点和宽高
center, radius = cv2.minEnclosingCircle(points)#绘制圆形 返回圆心 坐标和半径
hull = cv2.convexHull(points, clockwise, returnPoints)#绘制轮廓多边形，返回 轮廓数组
edges = cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)#返回计算后得出的边缘图像，是一个二值灰度图像。
lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)#返回线段端点坐标
circles = cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)#圆环检测 返回圆心坐标和半径





