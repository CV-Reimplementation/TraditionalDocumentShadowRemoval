
#include"TextDocShadRem2021.h" 

int main()
{  
	String strFolder = "imgs2/*.jpg"; //imgs1->HS imgs2->Test00 imgs3->Adobe
	vector<cv::String> imageFiles;
	glob(strFolder, imageFiles, false);

	char imgPath[100];
	char cSaveImg[150];

	size_t count = imageFiles.size();
	char savePath[100];
	cout << count << endl;
	if (count < 1)
	{
		cout << "Please give invalid folder path" << endl;
	}

	/*clock_t start;
	double duration;
	double sumtimes = 0;*/

	int allNum = 0;
	for (size_t i = 0; i < count; i++)
	{
		//读取图片，BGR三通道的
		Mat img = imread(imageFiles[i]); //直读取图片并返回Mat类型

		cout << imageFiles[i] << endl;
		if (img.empty())
		{
			cout << "Can not load the image file" << endl;
			return -1;
		}
		allNum++;

		int beginChar = imageFiles[i].find_first_of('\\');
		int endChar = imageFiles[i].find_first_of('.');
		string imgName = imageFiles[i].substr(beginChar + 1, (endChar - beginChar - 1));

		Mat result(img.size(), CV_8UC3, 3);
		Mat hlsImg(img.size(), CV_8UC3, 3); 
		Mat resultMerge(img.size(), CV_8UC3, Scalar(0, 0, 0));
		Mat bg(img.size(), CV_8UC3, Scalar(0, 0, 0));

		Mat gray(img.size(), CV_8UC1, Scalar(0));
		Mat binaryFromOriginalImage(img.size(), CV_8UC1, Scalar(0));
		Mat Unumbra(img.size(), CV_8UC1, Scalar(0));
		Mat umbra(img.size(), CV_8UC1, Scalar(0));

		Mat imgTemp(img.size(), CV_8UC3, Scalar(0, 0, 0));
		img.copyTo(imgTemp);

		int refLightRegionBgValue[3] = { 255 };
		double refUnLightRegionContrast[3] = { 1 };

		//start = clock();
		//1 转灰度
		cvtColor(img, gray, CV_BGR2GRAY);
		/*sprintf(cSaveImg, "results_gray/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, gray);*/

		//2 得到第一次去除阴影的图片，文字二值图， 阴影部分
		TDSR_Unumbra(gray, umbra, Unumbra, binaryFromOriginalImage);//gray
		/*sprintf(cSaveImg, "results_umbra/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, umbra);
		sprintf(cSaveImg, "results_unumbra/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, Unumbra);
		sprintf(cSaveImg, "results_binaryFromOriginal/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, binaryFromOriginalImage);*/

		//3 得到全局背景值， 阴影部分文字与非阴影部分文字的对比度
		TDSR_BgValue(img, umbra, binaryFromOriginalImage, refLightRegionBgValue, refUnLightRegionContrast, bg);//img
		/*sprintf(cSaveImg, "results_bg/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, bg);*/

		//4 融合字与背景
		TDSR_FuseBinary_and_GlobalBg(img, binaryFromOriginalImage,bg, result);
		/*sprintf(cSaveImg, "results/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, result);*/
	
		//5 文字对比度增强
		TDSR_ConvertHls(result, umbra, binaryFromOriginalImage, refUnLightRegionContrast, resultMerge);

		/*duration = (clock() - start) / (double)CLOCKS_PER_SEC;
		cout << duration << endl;
		sumtimes += duration;*/

		sprintf(cSaveImg, "results_merge/%s.jpg", imgName.c_str());
		imwrite(cSaveImg, resultMerge);

		waitKey(0);
	}
	/*sumtimes /= count;
	cout << sumtimes << endl;*/

	return 0;
}




 
