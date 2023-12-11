function [ sh,divided,msk,iter ] = maskMeanFiltcpp(im,iterations)
%INCREMENTALMASKMEANFILTOPTIMIZED Performs averaging of text pixels by non-text pixels
%   im : Input image
%   iterations : Number of iterations to run

% Binarizing
sensitivity2 = 0.55;
imgray = rgb2gray(im);
nbhood = 2*floor(size(imgray)/16)+1;
T = adaptthresh(imgray,sensitivity2,'NeighborhoodSize',nbhood,'ForegroundPolarity','dark');
msk = imbinarize(imgray,T);

% Dilation parameters
se = strel('disk',5);
prevmsk = logical(zeros(size(msk)));

iter = iterations;
for it = 1:iterations
    msk = ~imdilate(~msk,se); % Dilating to include borders pixels of text regions
    if isequal(prevmsk,msk)
        iter = it;
        break;
    end
    sh = maskMeanFilt(double(im),double(msk));
    divided = double(im)./double(sh);
    dividedgray = rgb2gray(divided); % Binarizing computed reflectance again
    T2 = adaptthresh(dividedgray,sensitivity2,'NeighborhoodSize',nbhood,'ForegroundPolarity','dark');
    prevmsk = msk;
    msk = imbinarize(dividedgray,T2);
end
msksh = imbinarize(rgb2gray(uint8(sh))); % Mask containing non-shadow region
mskg = msksh & msk; % Mask containing non-shadow and non-text region
% figure,imshow(mskg);
sh = double(sh);
mskIm = double(im).*double(cat(3,mskg,mskg,mskg));
gmean = reshape(sum(sum(mskIm)),1,3)/sum(sum(mskg)); % Mean of pixels which are not in shadow and are not text
sh(:,:,1) = sh(:,:,1)/gmean(1);
sh(:,:,2) = sh(:,:,2)/gmean(2);
sh(:,:,3) = sh(:,:,3)/gmean(3);
divided(:,:,1) = divided(:,:,1) * gmean(1);
divided(:,:,2) = divided(:,:,2) * gmean(2);
divided(:,:,3) = divided(:,:,3) * gmean(3);
end