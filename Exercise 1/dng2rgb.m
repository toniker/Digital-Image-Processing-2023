function [Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method)
% Apply White balance
mask = colormask(size(rawim,1), size(rawim,2), wbcoeffs, bayertype);
color_balanced_im = rawim .* mask;

% Demosaic
customRGB = custom_demosaic(color_balanced_im, bayertype, method);

% Color Space Transformation
XYZ2RGB = [[+3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]];

Clinear = customRGB;
Csrgb = Clinear .^ (1/2.2);
Csrgb = real(Csrgb);
Cxyz = apply_cmatrix(customRGB, XYZ2RGB);
Ccam = apply_cmatrix(customRGB, XYZ2Cam);

hsv = rgb2hsv(Csrgb);
hsv(:,:,1) = hsv(:,:,1) * 1.15;
hsv(:,:,2) = hsv(:,:,2) * 1.65;
% hsv(:,:,3) = hsv(:,:,3) * 1.2;
rgb = hsv2rgb(hsv);
imwrite(rgb, "test.jpg");

end
