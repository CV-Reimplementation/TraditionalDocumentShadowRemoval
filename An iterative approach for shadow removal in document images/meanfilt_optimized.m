clear all;close all;clc;

%% Input image
I = imread('');

%% Running algorithm
[sh,rf,msk,iter] = maskMeanFiltcpp(I,10);

%% Displaying results
figure,imshow(uint8(rf));
figure,imshow(sh);

%% Matching (to match output to a ground truth for benchmarking)
% gt = imread('');
% matched = matching(double(gt)/255,rf/255);
% figure,imshow(matched);