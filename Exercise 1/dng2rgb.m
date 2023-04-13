function [Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method)
% Apply White balance
mask = colormask(size(rawim,1), size(rawim,2), wbcoeffs, bayertype);
color_balanced_im = rawim .* mask;
% montage([color_balanced_im, rawim]);

% Interpolation
if (method == "linear")
    [rows, cols] = size(color_balanced_im);
    interpolated_im = zeros(rows, cols);
    for i = 2 : rows - 1
        for j = 2 : cols - 1
            interpolated_im(i,j) = (color_balanced_im(i-1,j) + color_balanced_im(i+1,j) + color_balanced_im(i,j-1) + color_balanced_im(i,j+1)) / 4;
        end
    end
elseif (method == "nearest")
    [rows, cols] = size(color_balanced_im);
    interpolated_im = zeros(rows, cols);
    for i = 2 : rows - 1
        for j = 2 : cols - 1
            interpolated_im(i,j) = color_balanced_im(i-1,j);
        end
    end
end

% Demosaic
customRGB = custom_demosaic(color_balanced_im, bayertype);
montage(customRGB);
% imshow(customRGB);
temp = uint16(color_balanced_im/max(color_balanced_im(:))*2^16);
builtinRGB = double(demosaic(temp,"rggb"))/2^16;
montage([customRGB, builtinRGB]);
% Color Space Transformation
XYZ2RGB = [[+3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]];

Clinear = 0;
Csrgb = Clinear .^ (1/2.2);
Cxyz = apply_cmatrix(interpolated_im, XYZ2RGB);
Ccam = apply_cmatrix(interpolated_im, XYZ2Cam);
end
