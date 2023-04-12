function [rawim, XYZ2Cam, wbcoeffs] = readdng(filename)
    obj = Tiff(filename ,'r');

    offsets = getTag(obj, 'SubIFD');
    setSubDirectory(obj, offsets(1));
    rawim = read(obj);
    meta_info = imfinfo (filename);
    
    % (x_origin , y_origin ) is the uper left corner of the useful part of the sensor and consequently of the array rawim
    y_origin = meta_info.SubIFDs{1}.ActiveArea(1) + 1;
    x_origin = meta_info.SubIFDs{1}.ActiveArea(2) + 1;
    
    % width and height of the image (the useful part of array rawim)
    width = meta_info.SubIFDs{1}.DefaultCropSize(1);
    height = meta_info.SubIFDs{1}.DefaultCropSize(2);
    
    blacklevel = meta_info.SubIFDs{1}.BlackLevel(1); % sensor value corresponding to black
    whitelevel = meta_info.SubIFDs{1}.WhiteLevel; % sensor value corresponding to white
    
    wbcoeffs = (meta_info.AsShotNeutral) .^ -1;
    wbcoeffs = wbcoeffs / wbcoeffs(2); % green channel will be left unchanged
    
    XYZ2Cam = meta_info.ColorMatrix2;
    XYZ2Cam = reshape(XYZ2Cam, 3, 3);
    
    rawim = double(rawim); % Cast array to doubles to keep floating point precision.
    rawim = rawim ./ (whitelevel - blacklevel); % Scale values to dynamic range of image.
    rawim = rawim + blacklevel; % Offset values to match the black level.
    rawim = max(0,min(rawim,1)); % Clip maximum values beyond the maximum white level.
    
    % imshow(rawim);
    close(obj);
end
