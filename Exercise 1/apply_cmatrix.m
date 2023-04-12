function corrected = apply_cmatrix(image, color_matrix)
% Applies CMATRIX to RGB input IM. Finds the appropriate weighting of the
% old color planes to form the new color planes, equivalent to but much
% more efficient than applying a matrix transformation to each pixel.
if size(image,3)~=3
    error('Apply cmatrix to RGB image only.')
end

r = color_matrix(1,1) * image(:,:,1)+color_matrix(1,2) * image(:,:,2) + color_matrix(1,3) * image(:,:,3);
g = color_matrix(2,1) * image(:,:,1)+color_matrix(2,2) * image(:,:,2) + color_matrix(2,3) * image(:,:,3);
b = color_matrix(3,1) * image(:,:,1)+color_matrix(3,2) * image(:,:,2) + color_matrix(3,3) * image(:,:,3);

corrected = cat(3,r,g,b);
end
