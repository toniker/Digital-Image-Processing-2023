function colormask = colormask(m, n, wb_coefficients, alignment)
% Makes a white-balance multiplicative mask for an image of size m-by-n
% with RGB white balance multipliers: wb_coefficients = [R_scale G_scale B_scale].
% alignment is a string indicating the Bayer arrangement
colormask = wb_coefficients(2)*ones(m,n); % Initialize to all green values
switch alignment
    case "bggr"
        colormask(2:2:end,2:2:end) = wb_coefficients(1); %r
        colormask(1:2:end,1:2:end) = wb_coefficients(3); %b
    case "gbrg"
        colormask(2:2:end,1:2:end) = wb_coefficients(1); %r
        colormask(1:2:end,2:2:end) = wb_coefficients(3); %b
    case "grbg"
        colormask(1:2:end,2:2:end) = wb_coefficients(1); %r
        colormask(1:2:end,2:2:end) = wb_coefficients(3); %b
    case "rggb"
        colormask(1:2:end,1:2:end) = wb_coefficients(1); %r
        colormask(2:2:end,2:2:end) = wb_coefficients(3); %b
end
end