filename = "RawImage.tiff";
[rawim, XYZ2Cam, wbcoeffs] = readdng(filename);

bayertype = "rggb";
method = "linear";
[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

% Write image files from the dng2rgb output
%imwrite(Csrgb, "output_rgb");
%imwrite(Clinear, "output_linear");
%imwrite(Cxyz, "output_xyz");
%imwrite(Ccam, "output_cam");

% Create histograms for R, G, B
%histogram_red = histogram(X);
%histogram_green= histogram(X);
%histogram_blue = histogram(X);
