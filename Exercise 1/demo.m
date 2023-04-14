filename = "RawImage.tiff";
[rawim, XYZ2Cam, wbcoeffs] = readdng(filename);

bayertype = "rggb";
method = "linear";
[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

% Write image files from the dng2rgb output
imwrite(Csrgb, "output_rgb.jpg");
imwrite(Clinear, "output_linear.jpg");
imwrite(Cxyz, "output_xyz.jpg");
imwrite(Ccam, "output_cam.jpg");

% Create histograms for R, G, B
hold on

[redCounts, redBins] = imhist(Csrgb(:,:,1));
[greenCounts, greenBins] = imhist(Csrgb(:,:,2));
[blueCounts, blueBins] = imhist(Csrgb(:,:,3));

stem(blueBins, blueCounts, ".-b")
stem(greenBins, greenCounts, ".-g")
stem(redBins, redCounts, ".-r")

hold off

