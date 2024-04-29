//Note: This code is created by Bingshu Wang , 2019.08.31
//We proposed an effective method to remove shadows from document images.
//This is only for academic exchange. 
// If one wants use it for commercial purpose, please contact us right now by yb77408@umac.mo or philipchen@um.edu.mo.  
// or  https://www.fst.um.edu.mo/en/staff/pchen.html. 

//If you try to use this code, please cite our paper 
// "An Effective Background Estimation Method For Shadows Removal Of Document Images"  
// accepted by ICIP2019.

#include"doc_shadow_removal.h"
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <iostream>
#include <algorithm>

int main(int argc, char** argv)
{

	// Check input
	// main ../dataset/Adobe/train/input/ ./
	if (argc != 3) {
		cout << "Usage: DocumentShadowRemoval.exe InputLocation OutputLocation" << endl; 
		getchar();
		return -1;
	}

	struct dirent *entry;
    DIR *dp;

    dp = opendir(argv[1]);
    if (dp == NULL) {
        perror("opendir: Path does not exist or could not be read.");
        return -1;
    }

	while ((entry = readdir(dp))) {
		string f_name = entry->d_name;
		transform(f_name.begin(), f_name.end(), f_name.begin(), ::toupper);

		if (f_name.find("JPG") != string::npos || f_name.find("PNG") != string::npos) {
			string src = argv[1];
			src += entry->d_name;

			string dst = argv[2];
			dst += entry->d_name;

			const int length1 = src.length();
  
			// declaring character array (+1 for null terminator)
			char* src_arr = new char[length1 + 1];
		
			// copying the contents of the
			// string to char array
			strcpy(src_arr, src.c_str());
			// ShadowRemover* sr = new ShadowRemover(src_arr);

			Mat img = imread(src_arr);

			Mat result(img.size(), CV_8UC3, 3);
	
			ShadowRemoval(img, result);

			
			const int length2 = dst.length();
  
			// declaring character array (+1 for null terminator)
			char* dst_arr = new char[length2 + 1];
		
			// copying the contents of the
			// string to char array
			strcpy(dst_arr, dst.c_str());
			
			imwrite(dst_arr, result);
		}
	}

	
	// Mat gt = imread("000_010gt.bmp");
	// Mat result(img.size(), CV_8UC3, 3);
	
	// ShadowRemoval(img, result);

	// imshow("img", img);
	// imshow("result",result);

	// double dMSE1 = CalulateOneImgMSE(result, gt);
	// double dMSE2 = CalulateOneImgMSE(img, gt);
	// double  error_ratio = sqrt(dMSE1) / sqrt(dMSE2);
	// cout << "  MSE1: " << dMSE1 << "    MSE2:"<< dMSE2<<"     Error_ratio:"<< error_ratio <<endl;

	// waitKey(0);
	closedir(dp);
	return 0;
}







 

 