#include "mexopencv.hpp"

using namespace std;
using namespace cv;

#define INITIAL_WINDOW_SIZE 11
#define MIN_PIXELS 25

int getPixel(Mat m,int i,int j){
    Scalar intensity = m.at<int>(i,j);
    return intensity.val[0];
}

Mat maskMeanFilt(Mat& im,Mat& mask){
    Mat intImageBlue,intImageGreen,intImageRed,intImageMask,intImageImg,divided,mask3d,imfiltered;
    int rmin,rmax,cmin,cmax,fl,w;
    double s1,s2,s3,c2;
    Mat sh = im.clone();
    vector<Mat> rgb(3),msk;
    
    msk.push_back(mask);msk.push_back(mask);msk.push_back(mask);
    merge(msk,mask3d);
    imfiltered = im.mul(mask3d);
    split(imfiltered,rgb);
    integral(rgb[0],intImageBlue);
    integral(rgb[1],intImageGreen);
    integral(rgb[2],intImageRed);
    integral(mask,intImageMask);
    Vec3b v;
    for(int i = 0;i<im.rows;i++){
        for(int j=0;j<im.cols;j++){
            Scalar intensity = mask.at<uchar>(i,j);
            if(intensity.val[0] == 0){
                fl = 0;
                w = INITIAL_WINDOW_SIZE;
                while(fl == 0){
                    rmin = max(i-w,0);rmax = min(i+w,im.rows-1);
                    cmin = max(j-w,0);cmax = min(j+w,im.cols-1);
                    
                    s1 = (double)(getPixel(intImageBlue,rmax+1,cmax+1) + getPixel(intImageBlue,rmin,cmin) - getPixel(intImageBlue,rmin,cmax+1) - getPixel(intImageBlue,rmax+1,cmin));
                    s2 = (double)(getPixel(intImageGreen,rmax+1,cmax+1) + getPixel(intImageGreen,rmin,cmin) - getPixel(intImageGreen,rmin,cmax+1) - getPixel(intImageGreen,rmax+1,cmin));
                    s3 = (double)(getPixel(intImageRed,rmax+1,cmax+1) + getPixel(intImageRed,rmin,cmin) - getPixel(intImageRed,rmin,cmax+1) - getPixel(intImageRed,rmax+1,cmin));
                    c2 = (double)(getPixel(intImageMask,rmax+1,cmax+1) + getPixel(intImageMask,rmin,cmin) - getPixel(intImageMask,rmin,cmax+1) - getPixel(intImageMask,rmax+1,cmin));
                    
                    if(c2 > MIN_PIXELS){
                        v[0] = (char)(s1/c2);v[1] = (char)(s2/c2);v[2] = (char)(s3/c2);
                        sh.at<Vec3b>(i,j) = v;
                        fl = 1;
                    }
                    else{
                        w += 1;
                    }
                }
            }
        }
    }
    return sh;
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    Mat image,mask;
    Mat inputImage = MxArray(prhs[0]).toMat();
    Mat inputMask = MxArray(prhs[1]).toMat();
    inputImage.convertTo(image,CV_8UC3);
    inputMask.convertTo(mask,CV_8UC1);
    Mat sh = maskMeanFilt(image,mask);
    plhs[0] = MxArray(sh);
}
