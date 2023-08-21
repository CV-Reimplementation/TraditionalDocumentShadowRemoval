#include"TextDocShadRem2021.h"


//得到第一次去除阴影的图片，文字二值图， 阴影部分
void TDSR_Unumbra(Mat& gray, Mat& umbra, Mat& outputMat, Mat& binary) {

	Mat rgbChannels = gray.clone();
	Mat element1 = getStructuringElement(MORPH_RECT, Size(11, 11));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(13, 13));
	dilate(rgbChannels, rgbChannels, element1, Point(-1, -1), 1, 0);
	medianBlur(rgbChannels, rgbChannels, 51);//中值滤波平滑（模糊）//51
	erode(rgbChannels, rgbChannels, element2, Point(-1, -1), 1, 0);
	Mat closeMat = rgbChannels;
	threshold(rgbChannels, umbra, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);//二值化

	//Mat calcMat = ~(closeMat - gray);
	Mat calcMat;
	absdiff(closeMat, gray, calcMat);
	calcMat = 255 - calcMat;

	Mat removeShadowMat;
	normalize(calcMat, removeShadowMat, 0, 200, NORM_MINMAX);//使用归一化将原来背景白色的改了和原来灰度图差不多的灰色
	outputMat = removeShadowMat;

	int blockSize = 31; //31
	adaptiveThreshold(removeShadowMat, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, 20); //20 //自适应二值化
}


//得到全局背景值， 阴影部分文字与非阴影部分文字的对比度
void TDSR_BgValue(Mat& img, Mat& shadingArea, Mat& binary, int refLightRegionBgValue[3], double refUnLightRegionContrast[3], Mat& bg) {

	int refUnLightRegionTextValue[3] = { 255 };
	int refLightRegionTextValue[3] = { 255 };
	int height = img.rows;
	int width = img.cols;

	double  sumLightRegionBg[3] = { 0 };  //统计非阴影区域整体亮度信息
	double  sumUnLightRegionText[3] = { 0 };
	double  sumLightRegionText[3] = { 0 };
	int numLightRegionBg = 0;
	int numUnLightRegionText = 0;
	int numLightRegionText = 0;

	for (int i = 0; i < height; i++)
	{
		uchar* puImg = img.ptr(i);
		uchar* puShadingArea = shadingArea.ptr(i);//整个阴影区域
		uchar* puBinary = binary.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadingArea[j] > 250)   //表示非阴影区域
			{
				if (puBinary[j] > 10) {
					//非阴影区域背景
					sumLightRegionBg[0] += puImg[3 * j];
					sumLightRegionBg[1] += puImg[3 * j + 1];
					sumLightRegionBg[2] += puImg[3 * j + 2];
					numLightRegionBg++;
				}
				else {
					sumLightRegionText[0] += puImg[3 * j];
					sumLightRegionText[1] += puImg[3 * j + 1];
					sumLightRegionText[2] += puImg[3 * j + 2];
					numLightRegionText++;
				}
			}
			if (puShadingArea[j] < 10)   //表示阴影区域
			{
				if (puBinary[j] < 10) {
					//阴影区域文字
					sumUnLightRegionText[0] += puImg[3 * j];
					sumUnLightRegionText[1] += puImg[3 * j + 1];
					sumUnLightRegionText[2] += puImg[3 * j + 2];
					numUnLightRegionText++;
				}
			}
		}
	}

	for (int k = 0; k < 3; k++)
	{
		if (numLightRegionBg > 0)
		{
			refLightRegionBgValue[k] = sumLightRegionBg[k] / numLightRegionBg;     //计算参考非阴影区域背景像素值

			refUnLightRegionTextValue[k] = sumUnLightRegionText[k] / numUnLightRegionText;
			refLightRegionTextValue[k] = sumLightRegionText[k] / numLightRegionText;
			refUnLightRegionContrast[k] = 1.0 * refUnLightRegionTextValue[k] / refLightRegionBgValue[k];
		}
	}

	vector<Mat> planes;
	cv::split(bg, planes);//生成全局背景图片
	planes[0].setTo(refLightRegionBgValue[0]);
	planes[1].setTo(refLightRegionBgValue[1]);
	planes[2].setTo(refLightRegionBgValue[2]);
	cv::merge(planes, bg);
}


//依据文字二值化图 和 全局背景进行融合
void TDSR_FuseBinary_and_GlobalBg(Mat& img, Mat& binary, Mat& bg, Mat& result) {
	img.copyTo(result);
	bg.copyTo(result, binary == 255);//背景部分
}


//文字对比度增强
void TDSR_ConvertHls(Mat& src, Mat& shadowImg, Mat& binary, double refUnLightRegionContrast[3], Mat& outputMat) {

	Mat img = src.clone();
	vector<double> m{ refUnLightRegionContrast[0] , refUnLightRegionContrast[1] , refUnLightRegionContrast[2] };
	multiply(img, m, img, 2.0);
	img.copyTo(outputMat);//对非阴影文字部分进行增强
	src.copyTo(outputMat, shadowImg == 0);//阴影部分不变
	src.copyTo(outputMat, binary == 255);//背景部分不变

	GaussianBlur(outputMat, outputMat, Size(3, 3), 1, 0);
}
