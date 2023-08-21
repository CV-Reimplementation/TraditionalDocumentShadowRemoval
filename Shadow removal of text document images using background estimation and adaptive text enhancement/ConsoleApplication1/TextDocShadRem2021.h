#pragma once
#ifndef _TexDocShadRem2021_H
#define _TexDocShadRem2021_H
#include"common.h"



/***********************************
* Function:从原图的灰度图像 得到阴影二值图，第一次去除阴影的图片，文字二值图
* Input:	gray：原图的灰度图像 
* Output:   umbra：阴影二值图 0：阴影 255：非阴影
*			unumbra：第一次去掉阴影的二值图
*			binary：文字二值图 0：文字 255：背景
* 
* return:void
* data:2022/09/29
***********************************/
void TDSR_Unumbra(Mat& gray, Mat& umbra, Mat& unumbra, Mat& binary);


/***********************************
* Function:从原图，阴影二值图和文字二值图 得到全局背景， 阴影部分文字与非阴影部分文字的对比度
* Input:	img：原图
* 			shadowImg：阴影二值图 0：阴影 255：非阴影
*			binary：文字二值图 0：文字 255：背景
* Output:	int refLightRegionBgValue[3]：全局背景值
*			double refUnLightRegionContrast[3]：阴影部分文字与非阴影部分文字的对比度
* 			bg：全局背景图
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_BgValue(Mat& img, Mat& shadowImg, Mat& binary, int refLightRegionBgValue[3], double refUnLightRegionContrast[3], Mat& bg);


/***********************************
* Function:依据文字二值化图 和 全局背景进行融合
* Input:	img：原图
*			binary：文字二值图 0：文字 255：背景
*			bg：全局背景图
* Output:	result：融合后的图像
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_FuseBinary_and_GlobalBg(Mat& img, Mat& binary, Mat& bg, Mat& result);


/***********************************
* Function:对融合后的图像进行文字对比度增强
* Input:	src：融合后的图像
*			binary：文字二值图 0：文字 255：背景
*			double refUnLightRegionContrast[3]：阴影部分文字与非阴影部分文字的对比度
* Output:	outputMat：文字对比度增强后的图像
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_ConvertHls(Mat& src, Mat& shadowImg, Mat& binary, double refUnLightRegionContrast[3], Mat& outputMat);


#endif