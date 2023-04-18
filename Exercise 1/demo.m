filename = "RawImage.tiff";
[rawim, XYZ2Cam, wbcoeffs] = readdng(filename);

bayertype = "rggb";
method = "linear";
[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

% Write image files from the dng2rgb output
imwrite(Csrgb, "MHC_"+method+"_"+bayertype+"_"+"rgb.jpg");
imwrite(Clinear, "MHC_"+method+"_"+bayertype+"_"+"linear.jpg");
imwrite(Cxyz, "MHC_"+method+"_"+bayertype+"_"+"xyz.jpg");
imwrite(Ccam, "MHC_"+method+"_"+bayertype+"_"+"cam.jpg");

% Create histograms for R, G, B
hold on

[redCounts, redBins] = imhist(Csrgb(:,:,1));
[greenCounts, greenBins] = imhist(Csrgb(:,:,2));
[blueCounts, blueBins] = imhist(Csrgb(:,:,3));

stem(blueBins, blueCounts, ".-b");
stem(greenBins, greenCounts, ".-g");
stem(redBins, redCounts, ".-r");

hold off

saveas(gcf, method+"_"+bayertype+"_"+"histogram.jpg");
