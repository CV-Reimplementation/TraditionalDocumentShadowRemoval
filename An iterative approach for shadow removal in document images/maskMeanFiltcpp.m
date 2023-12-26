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
function sh = maskMeanFilt(im, mask)
    % Define constants
    INITIAL_WINDOW_SIZE = 11;
    MIN_PIXELS = 25;

    % Initialize sh with the input image
    sh = im;

    % Perform the filter
    [rows, cols, ~] = size(im);
    for i = 1:rows
        for j = 1:cols
            if mask(i, j) == 0
                fl = 0;
                w = INITIAL_WINDOW_SIZE;
                while fl == 0
                    rmin = max(i-w, 1); rmax = min(i+w, rows);
                    cmin = max(j-w, 1); cmax = min(j+w, cols);

                    % Extract the local regions
                    localIm = im(rmin:rmax, cmin:cmax, :);
                    localMask = mask(rmin:rmax, cmin:cmax);

                    % Calculate the sum of the pixels in the local region
                    % weighted by the mask
                    localSum = sum(sum(bsxfun(@times, double(localIm), double(localMask)), 1), 2);
                    pixelCount = sum(localMask(:));

                    % Check if enough pixels are present
                    if pixelCount > MIN_PIXELS
                        % Calculate mean and assign it to the output pixel
                        v = localSum / pixelCount;
                        sh(i, j, :) = uint8(v);
                        fl = 1;
                    else
                        w = w + 1;
                    end
                end
            end
        end
    end
end