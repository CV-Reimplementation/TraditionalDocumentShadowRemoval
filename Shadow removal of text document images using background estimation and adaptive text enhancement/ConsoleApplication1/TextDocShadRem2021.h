#pragma once
#ifndef _TexDocShadRem2021_H
#define _TexDocShadRem2021_H
#include"common.h"



/***********************************
* Function:��ԭͼ�ĻҶ�ͼ�� �õ���Ӱ��ֵͼ����һ��ȥ����Ӱ��ͼƬ�����ֶ�ֵͼ
* Input:	gray��ԭͼ�ĻҶ�ͼ�� 
* Output:   umbra����Ӱ��ֵͼ 0����Ӱ 255������Ӱ
*			unumbra����һ��ȥ����Ӱ�Ķ�ֵͼ
*			binary�����ֶ�ֵͼ 0������ 255������
* 
* return:void
* data:2022/09/29
***********************************/
void TDSR_Unumbra(Mat& gray, Mat& umbra, Mat& unumbra, Mat& binary);


/***********************************
* Function:��ԭͼ����Ӱ��ֵͼ�����ֶ�ֵͼ �õ�ȫ�ֱ����� ��Ӱ�������������Ӱ�������ֵĶԱȶ�
* Input:	img��ԭͼ
* 			shadowImg����Ӱ��ֵͼ 0����Ӱ 255������Ӱ
*			binary�����ֶ�ֵͼ 0������ 255������
* Output:	int refLightRegionBgValue[3]��ȫ�ֱ���ֵ
*			double refUnLightRegionContrast[3]����Ӱ�������������Ӱ�������ֵĶԱȶ�
* 			bg��ȫ�ֱ���ͼ
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_BgValue(Mat& img, Mat& shadowImg, Mat& binary, int refLightRegionBgValue[3], double refUnLightRegionContrast[3], Mat& bg);


/***********************************
* Function:�������ֶ�ֵ��ͼ �� ȫ�ֱ��������ں�
* Input:	img��ԭͼ
*			binary�����ֶ�ֵͼ 0������ 255������
*			bg��ȫ�ֱ���ͼ
* Output:	result���ںϺ��ͼ��
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_FuseBinary_and_GlobalBg(Mat& img, Mat& binary, Mat& bg, Mat& result);


/***********************************
* Function:���ںϺ��ͼ��������ֶԱȶ���ǿ
* Input:	src���ںϺ��ͼ��
*			binary�����ֶ�ֵͼ 0������ 255������
*			double refUnLightRegionContrast[3]����Ӱ�������������Ӱ�������ֵĶԱȶ�
* Output:	outputMat�����ֶԱȶ���ǿ���ͼ��
*
* return:void
* data:2022/09/29
***********************************/
void TDSR_ConvertHls(Mat& src, Mat& shadowImg, Mat& binary, double refUnLightRegionContrast[3], Mat& outputMat);


#endif