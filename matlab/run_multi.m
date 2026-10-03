function run_multi(input_path)

% =========================
% CREATE DATA FOLDER
% =========================
if ~exist('data','dir')
    mkdir('data');
end

% =========================
% READ IMAGE
% =========================
img = imread(input_path);
img = im2double(img);

if size(img,1) > 600
    img = imresize(img,0.5);
end

% =========================
% PREPROCESSING
% =========================
if size(img,3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

gray = medfilt2(gray,[5 5]);
gray = adapthisteq(gray,'ClipLimit',0.02);

% =========================
% SUPER RESOLUTION
% =========================
low_res = imresize(gray,0.4,'bicubic');
low_res_up = imresize(low_res,size(gray),'bicubic');

blur1 = imgaussfilt(low_res_up,1);
blur2 = imgaussfilt(low_res_up,2);

detail = (low_res_up - blur1) + (blur1 - blur2);

edges = edge(gray,'canny');
edges = im2double(edges);

SR = low_res_up + 0.6*edges + 0.4*detail;
SR = mat2gray(SR);
SR = imsharpen(SR,'Radius',2,'Amount',1.8);

% =========================
% EXTRA PROCESSING
% =========================
histeq_img = histeq(gray);
edge_sobel = edge(gray,'sobel');
edge_prewitt = edge(gray,'prewitt');

bw = imbinarize(gray);
se = strel('disk',3);
morph = imopen(bw,se);

fft_img = fftshift(fft2(gray));
fft_display = log(1 + abs(fft_img));

% =========================
% SAVE OUTPUTS
% =========================
imwrite(SR,'data/SR.png');
imwrite(edges,'data/edges.png');
imwrite(edge_sobel,'data/sobel.png');
imwrite(edge_prewitt,'data/prewitt.png');
imwrite(histeq_img,'data/histeq.png');
imwrite(bw,'data/bw.png');
imwrite(morph,'data/morph.png');
imwrite(mat2gray(fft_display),'data/fft.png');

heatmap = ind2rgb(im2uint8(SR),jet(256));
imwrite(heatmap,'data/heatmap.png');

disp('Processing Complete');

end