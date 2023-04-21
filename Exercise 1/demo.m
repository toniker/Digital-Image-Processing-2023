filename = "RawImage.tiff";
[rawim, XYZ2Cam, wbcoeffs] = readdng(filename);

bayertype = "rggb";
method = "linear";
[Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype, method);

% Write image files from the dng2rgb output
imwrite(Csrgb, method+"_"+bayertype+"_"+"rgb.jpg");
imwrite(Clinear, method+"_"+bayertype+"_"+"linear.jpg");
imwrite(Cxyz, method+"_"+bayertype+"_"+"xyz.jpg");
imwrite(Ccam, method+"_"+bayertype+"_"+"cam.jpg");

% Create RGB histograms for Linear
hold on

[redCounts, redBins] = imhist(Clinear(:,:,1));
[greenCounts, greenBins] = imhist(Clinear(:,:,2));
[blueCounts, blueBins] = imhist(Clinear(:,:,3));

stem(blueBins, blueCounts, ".-b");
stem(greenBins, greenCounts, ".-g");
stem(redBins, redCounts, ".-r");

hold off

saveas(gcf, method+"_"+bayertype+"_linear_histogram.jpg");
clf;

% Create RGB histograms for Csrgb
hold on

[redCounts, redBins] = imhist(Csrgb(:,:,1));
[greenCounts, greenBins] = imhist(Csrgb(:,:,2));
[blueCounts, blueBins] = imhist(Csrgb(:,:,3));

stem(blueBins, blueCounts, ".-b");
stem(greenBins, greenCounts, ".-g");
stem(redBins, redCounts, ".-r");

hold off

saveas(gcf, method+"_"+bayertype+"_rgb_histogram.jpg");
clf;

% Create RGB histograms for XYZ
hold on

[redCounts, redBins] = imhist(Cxyz(:,:,1));
[greenCounts, greenBins] = imhist(Cxyz(:,:,2));
[blueCounts, blueBins] = imhist(Cxyz(:,:,3));

stem(blueBins, blueCounts, ".-b");
stem(greenBins, greenCounts, ".-g");
stem(redBins, redCounts, ".-r");

hold off

saveas(gcf, method+"_"+bayertype+"_xyz_histogram.jpg");
clf;

% Create RGB histograms for Cam
hold on

[redCounts, redBins] = imhist(Ccam(:,:,1));
[greenCounts, greenBins] = imhist(Ccam(:,:,2));
[blueCounts, blueBins] = imhist(Ccam(:,:,3));

stem(blueBins, blueCounts, ".-b");
stem(greenBins, greenCounts, ".-g");
stem(redBins, redCounts, ".-r");

hold off

saveas(gcf, method+"_"+bayertype+"_cam_histogram.jpg");
