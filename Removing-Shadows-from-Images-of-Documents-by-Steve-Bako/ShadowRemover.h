#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/photo/photo.hpp>
#include <cmath>
#include <random>

using namespace cv;
using namespace cv::ml;
using namespace std;

#define BUFFER_SIZE 1000
#define USE_SAMPLING 1

class ShadowRemover {

public:

	// Constructors and destructor
	ShadowRemover(char* in);
	~ShadowRemover();

	// Primary function of the ShadowRemover class. Start here
	void RemoveShadow(char* out);

	// Returns the block centered around (x,y). 
	bool GetBlock(int x, int y, Mat& block, Mat& dsImage);

	// Performs the local clustering using GMM at the block and saves the result in clusterMu
	void ClusterBlock(Mat& block, Mat& clusterMu, int* randInd);

	// Identifies the local cluster belonging to the background and saves it in the shadow map
	// Currently uses max cluster mean.
	void CalculatePaperStatistics(int x, int y, Mat& clusterMu);

	// Finds the block that best represents the background region. Used for constructing the
	// shadow map (gain map)
	void FindReferenceIndex(int& refInd, Mat& dsImage, Vec3f& ref);

	// Upsample shadow map by stride
	void UpsampleShadowMap();

	// Divide each local region by the reference
	void NormalizeShadowMap(int refIndex, Vec3f& ref);

	// Apply per pixel gain to do the actual shadow removal
	void ApplyShadowMap();

	// Save the output and the shadow map
	void SaveResults(char* out);

	// Converts x and y index to access downsampled images (xhat = x and
	// yhat = y when stride is 1)
	void ConvertIndex(int x, int y, int& xHat, int& yHat);

	// Wrappers for opencv utilities to read, write, and display images
	int ReadImage(Mat& img, char* filename);
	void DisplayImage(Mat img);
	void WriteImage(Mat img, char* filename, bool isNormalized = false);
	void SaveTiming(float val, char* out);

	// Getters
	Mat* GetImage();
	Mat* GetShadowMap();
	int GetWidth();
	int GetHeight();
	int GetChannels();
	int GetStride();
	int GetBlockSize();
	int GetNumOfClusters();
	int GetNumOfClustersRef();
	int GetMaxIters();
	float GetEps();

	// Setters
	void SetImage(Mat* image);
	void SetShadowMap(Mat* shadowMap);
	void SetWidth(int w);
	void SetHeight(int h);
	void SetChannels(int c);
	void SetStride(int s);
	void SetBlockSize(int b);
	void SetNumOfClusters(int c);
	void SetNumOfClustersRef(int c);
	void SetMaxIters(int iters);
	void SetEps(float eps);

private:

	// Input image, shadow map (gain map), and mask of image regions
	Mat* image;
	Mat* shadowMap;

	// Full width, height, and number of channels
	int width;
	int height;
	int channels;

	// Number of pixels to skip when performing local analysis
	int stride;

	// Size of overlapping blocks in local analysis
	int blockSize;

	// Number of clusters used for local analysis (i.e., 2)
	int numOfClusters;

	// Number of clusters used for global analysis (i.e., 3)
	int numOfClustersRef;

	// Maximum number of iterations and epsilon threshold used as stopping condition for GMM clustering
	int maxIters;
	float emEps;

	// Amount of downsampling to be used on the original image (for speedup)
	float dsFactor;

	// Number of local and global samples in the block and image, respectively (Default is 150 and 1000)
	int numOfLocalSamples;
	int numOfGlobalSamples;

};