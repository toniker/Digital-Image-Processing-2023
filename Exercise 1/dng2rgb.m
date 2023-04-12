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

% Color Space Transformation
XYZ2RGB = [[+3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]];

% Csrgb = Clinear .^ (1/2.2);
Csrgb = 0;
Clinear = 0;
Cxyz = 0;
Ccam = 0;
end