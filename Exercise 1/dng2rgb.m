function [Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method)
% Apply White balance
mask = colormask(size(rawim,1), size(rawim,2), wbcoeffs, bayertype);
color_balanced_im = rawim .* mask;

% Demosaic
customRGB = custom_demosaic(color_balanced_im, bayertype);
image(customRGB * 2);
temp = uint16(color_balanced_im/max(color_balanced_im(:))*2^16);
builtinRGB = double(demosaic(temp,"rggb"))/2^16;

% Interpolation
[rows, cols] = size(customRGB);
interpolated_im = zeros(rows, cols);
if (method == "linear")
    for i = 2 : rows - 1
        for j = 2 : cols - 1
            interpolated_im(i,j) = (customRGB(i-1,j) + customRGB(i+1,j) + customRGB(i,j-1) + customRGB(i,j+1)) / 4;
        end
    end
elseif (method == "nearest")
    for i = 2 : rows - 1
        for j = 2 : cols - 1
            interpolated_im(i,j) = customRGB(i-1,j);
        end
    end
end

% Color Space Transformation
XYZ2RGB = [[+3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]];

Clinear = 0;
Csrgb = Clinear .^ (1/2.2);
Cxyz = apply_cmatrix(interpolated_im, XYZ2RGB);
Ccam = apply_cmatrix(interpolated_im, XYZ2Cam);
end
