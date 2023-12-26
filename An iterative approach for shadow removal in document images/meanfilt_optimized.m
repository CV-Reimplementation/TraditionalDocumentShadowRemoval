% clear all;close all;clc;

%% Input image
% I = imread('rect_1.jpg');

%% Running algorithm


%% Displaying results
% figure,imshow(uint8(rf));
% figure,imshow(sh);

%% Matching (to match output to a ground truth for benchmarking)
% gt = imread('');
% matched = matching(double(gt)/255,rf/255);
% figure,imshow(matched);

clc;
clear all;
close all;
images_dir='input';
listing = cat(1, dir(fullfile(images_dir, '*.*g')));
% The final output will be saved in this directory:
results_dir = fullfile(images_dir, 'Result');
% Preparations for saving resultss.
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

n_iter=10;
for i_img = 1:1:length(listing)
    im = imread(fullfile(images_dir,listing(i_img).name));  

    [sh, rf, msk, iter] = maskMeanFiltcpp(im, n_iter);
    
    imwrite(sh, fullfile(results_dir, listing(i_img).name));

end