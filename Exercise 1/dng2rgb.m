function [Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method)
%% White balance
mask = colormask(size(rawim,1), size(rawim,2), wbcoeffs, bayertype);
color_balanced_im = rawim .* mask;

%% Demosaic
Clinear = custom_demosaic(color_balanced_im, bayertype, method);

% Use built in demosaic for comparison
imwrite((double(demosaic(uint16(color_balanced_im * 65535), bayertype)) / 65535) .^(1/2.2), "builtInDemosaic.jpg");

%% Color Space
XYZ2RGB = [[+3.2406; -1.5372; -0.4986], [-0.9689; +1.8758; +0.0415], [+0.0557; -0.2040; +1.0570]];
RGB2XYZ = inv(XYZ2RGB);

% Gamma Correction
Csrgb = real(Clinear .^ (1/2.2));

if (bayertype == "bggr" || bayertype == "rggb")
    % Convert to Hue, Saturation, Value
    hsv = rgb2hsv(Csrgb);

    % Hue shift
    hsv(:,:,1) = hsv(:,:,1) * 1.15;
    % Saturation boost
    hsv(:,:,2) = hsv(:,:,2) * 1.65;

    % Convert to RGB
    Csrgb = hsv2rgb(hsv);
end

% Color Space Transformation
Cxyz = apply_cmatrix(Csrgb, RGB2XYZ);
Ccam = apply_cmatrix(Cxyz, XYZ2Cam);

end
