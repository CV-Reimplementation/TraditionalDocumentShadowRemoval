#include "ShadowRemover.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

//*****************************CONSTRUCTOR/DESTRUCTOR**********************************//

ShadowRemover::ShadowRemover(char* in) {

	// Read in input image and set info
	image = new Mat();
	assert(ReadImage(*image, in));

	cout << in << endl;

	width = image->cols;
	height = image->rows;
	channels = image->channels();

	// Default algorithmic parameters
	stride = 20; // Number of pixels to skip when performing local analysis
	blockSize = 21; // Size of overlapping blocks in local analysis
	numOfClusters = 3; // Number of clusters used for local analysis
	numOfClustersRef = 3; // Number of clusters used for global analysis  
	maxIters = 100; // Maximum number of iterations used as stopping condition for GMM clustering. 
	emEps = 0.1f; // Epsilon threshold used as stopping condition for GMM clustering.
	dsFactor = 1.0f; // No downsampling is done
	numOfLocalSamples = 150; // Number of samples to take in each block (for local statistics)
	numOfGlobalSamples = 1000; // Number of samples to take throughout entire image (for global statistics)

	// Initialize shadow map
	int sHeight, sWidth;
	
	shadowMap = new Mat(height, width, CV_32FC3, CV_RGB(-1, -1, -1));
	resize(*shadowMap, *shadowMap, Size(0, 0), dsFactor, dsFactor, INTER_LANCZOS4);
	ConvertIndex(shadowMap->cols, shadowMap->rows, sWidth, sHeight);
	resize(*shadowMap, *shadowMap, Size(sWidth, sHeight));

}

ShadowRemover::~ShadowRemover() {
	delete image;
	delete shadowMap;
}

void ShadowRemover::RemoveShadow(char* out) {

	// Set up data structures for each thread (openMP)
	int threadCount = omp_get_max_threads();
	Mat* blockList = new Mat[threadCount];
	for (int i = 0; i < threadCount; i++) {
		blockList[i] = Mat(height, width, CV_32FC3, CV_RGB(0, 0, 0));
	}
	Mat dsMask = Mat(shadowMap->rows, shadowMap->cols, CV_8UC1, Scalar(0));

	// Start clustering

	// Randomly sample each block
	int* randInd = new int[numOfLocalSamples];
	int size = blockSize * blockSize; // TODO: use size_t
	vector<int> freeIndexes;
	for (int i = 0; i < size; i++) {
		freeIndexes.push_back(i);
	}
	int count = 0;
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, size);
	while (count < numOfLocalSamples) {
		int indexCandidate = distribution(generator);
		vector<int>::iterator it = std::find(freeIndexes.begin(), freeIndexes.end(), indexCandidate);
		if (it != freeIndexes.end()) {
			randInd[count] = indexCandidate;
			freeIndexes.erase(it);
			count++;
		}
	}

	// Downsample image (Default has no downsampling)
	Mat dsImage;
	resize(*image, dsImage, Size(0, 0), dsFactor, dsFactor, INTER_NEAREST);
	width = dsImage.cols;
	height = dsImage.rows;

	// Loop through pixels in image (with openMP)
	#pragma omp parallel 
	{
		#pragma omp for schedule(dynamic) nowait 
		for (int i = 0; i < height; i += stride) {
			for (int j = 0; j < width; j += stride) {

				// Get current block
				int threadNum = omp_get_thread_num();
				Mat& curBlock = blockList[threadNum];
				if (GetBlock(j, i, curBlock, dsImage)) {

					// Cluster pixel intensities
					Mat curMu;
					vector<Mat> listOfCovs;
					ClusterBlock(curBlock, curMu, randInd);

					// Find paper mu of current block and update global matrix
					CalculatePaperStatistics(j, i, curMu);

				}

			}
		}
	}

	// Clean up
	delete[] randInd;
	delete[] blockList;

	int refIndex = -1;
	Vec3f ref;
	FindReferenceIndex(refIndex, dsImage, ref);
	width = image->cols;
	height = image->rows;

	// Filter and upsample shadow map to match the resolution of the image
	medianBlur(*shadowMap, *shadowMap, 3);
	GaussianBlur(*shadowMap, *shadowMap, Size(3, 3), 2.5f);
	Mat dsShadowMap = *shadowMap;
	UpsampleShadowMap();

	// Generate shadow map (gain map) by comparing each block in input 
	// to the reference block
	NormalizeShadowMap(refIndex, ref);

	// Apply shadow map to original input to remove shadow
	ApplyShadowMap();

	// Save shadow-removed result and shadow map
	cout << out << endl;
	SaveResults(out);
}

bool ShadowRemover::GetBlock(int x, int y, Mat& block, Mat& dsImage) {

	// Find bounds around center pixel
	int halfBlock = (int) floorf(float(blockSize) / 2.0f);
	int minX = max(0, x - halfBlock);
	int maxX = min(width - 1, x + halfBlock);
	int minY = max(0, y - halfBlock);
	int maxY = min(height - 1, y + halfBlock);
	int deltaY = maxY - minY + 1;
	int deltaX = maxX - minX + 1;

	if (block.rows != deltaY || block.cols != deltaX) {
		block = Mat(deltaY, deltaX, CV_32FC3, CV_RGB(0, 0, 0));
	}

	// Copy intensities to block
	int bX = 0;
	int bY = 0;
	for (int i = minY; i <= maxY; i++) {
		for (int j = minX; j <= maxX; j++) {
			for (int k = 0; k < channels; k++) {
				block.at<Vec3f>(bY, bX)[k] = dsImage.at<Vec3f>(i, j)[k];
			}
			bX++;
		}
		bX = 0;
		bY++;
	}
	
	return true;

}

void ShadowRemover::ClusterBlock(Mat& block, Mat& clusterMu, int* randInd) {

	// Set up expectation maximization model
	Ptr<EM> emModel = EM::create();
	emModel->setClustersNumber(numOfClusters);
	emModel->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
	emModel->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, emEps));

	// Cluster block with k means initializer
	Mat samples;
	if (block.rows * block.cols == blockSize * blockSize) {
		Mat tmp(numOfLocalSamples, 1, CV_32FC3, CV_RGB(-1, -1, -1));
		for (int i = 0; i < numOfLocalSamples; i++) {
			assert(randInd[i] >= 0 && randInd[i] < block.rows * block.cols);
			tmp.at<Vec3f>(i) = block.at<Vec3f>(randInd[i]);
		}
		samples = tmp.reshape(1);
	}
	else {
		samples = block.reshape(0, block.rows * block.cols);
		samples = samples.reshape(1);
	}
	emModel->trainEM(samples);
	
	clusterMu = emModel->getMeans();
	clusterMu = clusterMu.reshape(channels);
	clusterMu.convertTo(clusterMu, CV_32FC3);

}

void ShadowRemover::CalculatePaperStatistics(int x, int y, Mat& clusterMu) {

	// Choose the highest cluster mean as the paper reference for this local region
	int sX, sY;
	ConvertIndex(x, y, sX, sY);
	Vec3f& shadowVec = shadowMap->at<Vec3f>(sY, sX);
	double maxSum = 0;
	for (int i = 0; i < numOfClusters; i++) {
		double muSum = 0;
		for (int k = 0; k < channels; k++) {
			muSum += clusterMu.at<Vec3f>(i)[k];
		}
		if (muSum > maxSum) {
			maxSum = muSum;
			for (int k = 0; k < channels; k++) {
				shadowVec[k] = clusterMu.at<Vec3f>(i)[k];
			}
		}
	}

}

void ShadowRemover::FindReferenceIndex(int& refIndex, Mat& dsImage, Vec3f& ref) {

	// Set up expectation maximization model
	Ptr<EM> emModel = EM::create();
	emModel->setClustersNumber(numOfClustersRef);
	emModel->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
	emModel->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, maxIters, emEps)); 

	// Cluster block with k means initializer
	Mat samples;

#if USE_SAMPLING

		Mat tmp(numOfGlobalSamples, 1, CV_32FC3, CV_RGB(-1, -1, -1));
		int* randInd = new int[numOfGlobalSamples];
		int size = width * height; // TODO: Use size_t
		vector<int> freeIndexes;
		for (int i = 0; i < size; i++) {
			freeIndexes.push_back(i);
		}
		int count = 0;
		int maxIndexCandidiate = -1;
		int delta = size / numOfGlobalSamples;
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0, size);
		while (count < numOfGlobalSamples) {
			int indexCandidate = distribution(generator);
			vector<int>::iterator it = std::find(freeIndexes.begin(), freeIndexes.end(), indexCandidate);
			if (it != freeIndexes.end()) {
				randInd[count] = indexCandidate;
				freeIndexes.erase(it);
				count++;
			}
		}
		for (int i = 0; i < numOfGlobalSamples; i++) {
			tmp.at<Vec3f>(i) = image->at<Vec3f>(randInd[i]);
		}
		delete[] randInd;
		samples = tmp.reshape(1);

#else
		
		samples = dsImage.reshape(0, width * height);	
		samples = samples.reshape(1);
	
#endif

	emModel->trainEM(samples);

	// Get the cluster means
	Mat clusterMu = emModel->getMeans();
	clusterMu = clusterMu.reshape(channels);
	clusterMu.convertTo(clusterMu, CV_32FC3);

	// Get cluster variances
	int maxInd = -1;
	double curMax = -1;
	for (int i = 0; i < numOfClustersRef; i++) {
		double muMag = 0;
		for (int k = 0; k < channels; k++) {
			muMag += clusterMu.at<Vec3f>(i)[k];
		}
		if (muMag > curMax) {
			curMax = muMag;
			maxInd = i;
		}
	}

	assert(maxInd != -1 && maxInd < numOfClustersRef);
	
	// Find the closest actual value to the cluster to choose as reference 
	// TODO: stop earlier once threshold is met?
	ref = clusterMu.at<Vec3f>(maxInd);
	float curMin = std::numeric_limits<float>::max();
	refIndex = -1;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3f curVal = dsImage.at<Vec3f>(i, j);
			float curMag = 0;
			for (int k = 0; k < channels; k++) {
				float diff = curVal[k] - ref[k];
				curMag += diff * diff;
			}
			if (curMag < curMin) {
				curMin = curMag;
				refIndex = j + i * width;
			}
		}
	}

}

void ShadowRemover::UpsampleShadowMap() {

	resize(*shadowMap, *shadowMap, Size(width, height), 0, 0, INTER_LANCZOS4);

}

void ShadowRemover::NormalizeShadowMap(int refIndex, Vec3f& ref) {

	assert(shadowMap->rows == height && shadowMap->cols == width);
	assert(refIndex >= 0 && refIndex < width * height);

	ref = shadowMap->at<Vec3f>(refIndex);
	
	// Divide each local paper intensity by the global reference
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3f& curShadowVec = shadowMap->at<Vec3f>(i, j);
			for (int k = 0; k < channels; k++) {

				curShadowVec[k] /= ref[k];

				// Clamp negative and zero values to a small number
				if (curShadowVec[k] <= 0) {
					curShadowVec[k] = 1.0e-6f;
				}	
			}
		}
	}

}

void ShadowRemover::ApplyShadowMap() {
	
	// Loop through all the pixels and divide by inverse gain
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3f invGain = shadowMap->at<Vec3f>(i, j);
			Vec3f& color = image->at<Vec3f>(i, j);
			for (int k = 0; k < channels; k++) {
				color[k] /= invGain[k];
			}
		}
	}

}

void ShadowRemover::SaveResults(char* out) {

	// char buff[BUFFER_SIZE];

	// cout << buff << endl;

	WriteImage(*image, out);
	// WriteImage(buff, *image);

	// WriteImage(*shadowMap, buff, true);

}

void ShadowRemover::ConvertIndex(int x, int y, int& xHat, int& yHat) {

	// Convert from original resolution to downsampled size (downsampled based on stride)
	xHat = (int)floor((x - 1) / float(stride)) + 1;
	yHat = (int)floor((y - 1) / float(stride)) + 1;

}

//*****************************OPENCV UTILITIES**********************************//

int ShadowRemover::ReadImage(Mat& img, char* filename) {

	img = imread(filename, CV_LOAD_IMAGE_COLOR);

	if (!img.data) {
		cout << "Could not open or find the image " << filename << std::endl; 
		return 0;
	}

	img.convertTo(img, CV_32FC3);

	return 1;

}

void ShadowRemover::DisplayImage(Mat img) {

	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", img);

	waitKey(0);
}

void ShadowRemover::WriteImage(Mat img, char* filename, bool isNormalized) {

	if (!img.data) {
		cout << "Image " << filename << " does not have data" << std::endl;
		getchar();
		exit(-1);
	}

	if (isNormalized) {
		img *= 255;
	}

	imwrite(filename, img);

}

void ShadowRemover::SaveTiming(float val, char* out) {

	// char name[BUFFER_SIZE];
	// FILE* fp;
	// sprintf(name, "%s_Timing.txt", out);

	// fopen_s(&fp, name, "wt");

	// if (!fp)
	// {
	// 	fprintf(stderr, "ERROR: Could not open dat file %s\n", name);
	// 	getchar();
	// 	exit(-1);
	// }

	// fprintf(fp, "%5.3f\n", val);
	// fclose(fp);

}

//*****************************GETTERS/SETTERS**********************************//

Mat* ShadowRemover::GetImage() {
	return image;
}

Mat* ShadowRemover::GetShadowMap() {
	return shadowMap;
}

int ShadowRemover::GetWidth() {
	return width;
}

int ShadowRemover::GetHeight() {
	return height;
}

int ShadowRemover::GetChannels() {
	return channels;
}

int ShadowRemover::GetStride() {
	return stride;
}

int ShadowRemover::GetBlockSize() {
	return blockSize;
}

int ShadowRemover::GetNumOfClusters() {
	return numOfClusters;
}

int ShadowRemover::GetNumOfClustersRef() {
	return numOfClustersRef;
}

int ShadowRemover::GetMaxIters() {
	return maxIters;
}

float ShadowRemover::GetEps() {
	return emEps;
}

void ShadowRemover::SetImage(Mat* image) {
	this->image = image;
}

void ShadowRemover::SetShadowMap(Mat* shadowMap) {
	this->shadowMap = shadowMap;
}

void ShadowRemover::SetWidth(int w) {
	this->width = w;
}

void ShadowRemover::SetHeight(int h) {
	this->height = h;
}

void ShadowRemover::SetChannels(int c) {
	this->channels = c;
}

void ShadowRemover::SetStride(int s) {
	this->stride = s;
}

void ShadowRemover::SetBlockSize(int b) {
	this->blockSize = b;
}

void ShadowRemover::SetNumOfClusters(int c) {
	this->numOfClusters = c;
}

void ShadowRemover::SetNumOfClustersRef(int c) {
	this->numOfClustersRef = c;
}

void ShadowRemover::SetMaxIters(int iters) {
	this->maxIters = iters;
}

void ShadowRemover::SetEps(float eps) {
	this->emEps = eps;
}