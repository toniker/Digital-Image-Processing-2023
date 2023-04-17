function RGB = custom_demosaic(input_image, bayertype, method)
RGB = 0; %#ok<NASGU>
if method == "nearest"
    switch bayertype
        case "bggr"
            RGB = internal_nearest_bggr_demosaic(input_image);
        case "gbrg"
            RGB = internal_nearest_gbrg_demosaic(input_image);
        case "grbg"
            RGB = internal_nearest_grbg_demosaic(input_image);
        case "rggb"
            RGB = internal_nearest_rggb_demosaic(input_image);
        otherwise
            error("Invalid bayertype");
    end
elseif method == "linear"
    switch bayertype
        case "bggr"
            RGB = internal_bilinear_bggr_demosaic(input_image);
        case "gbrg"
            RGB = internal_bilinear_gbrg_demosaic(input_image);
        case "grbg"
            RGB = internal_bilinear_grbg_demosaic(input_image);
        case "rggb"
            RGB = internal_bilinear_rggb_demosaic(input_image);
        otherwise
            error("Invalid bayertype");
    end
else
    error("Invalid method");
end
end

function J = internal_bilinear_rggb_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Blue pixel
            J(i,j,:) = [(I(i-1,j-1)+I(i+1,j+1)+I(i-1,j+1)+I(i+1,j-1))/4, (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, I(i,j)];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            J(i,j,:) = [(I(i-1,j)+I(i+1,j))/2, I(i,j), (I(i,j-1)+I(i,j+1))/2];
        else % Red pixel
            J(i,j,:) = [I(i,j), (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, (I(i-1,j-1)+I(i+1,j+1)+I(i-1,j+1)+I(i+1,j-1))/4];
        end
    end
end
end

function J = internal_nearest_rggb_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Blue pixel
            % J(i,j,:) = 
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            % J(i,j,:) = 
        else % Red pixel
            % J(i,j,:) = 
        end
    end
end
end

function J = internal_bilinear_bggr_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Red pixel
            J(i,j,:) = [I(i,j), (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, (I(i,j-1)+I(i,j+1))/2];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            J(i,j,:) = [(I(i-1,j)+I(i+1,j))/2, I(i,j), (I(i,j-1)+I(i,j+1))/2];
        else % Blue pixel
            J(i,j,:) = [(I(i,j-1)+I(i,j+1))/2, (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, I(i,j)];
        end
    end
end
end

function J = internal_nearest_bggr_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Red pixel
            % J(i,j,:) =
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            % J(i,j,:) =
        else % Blue pixel
            % J(i,j,:) =
        end
    end
end
end

function J = internal_bilinear_gbrg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if mod(i+j,2) == 0 % Green pixel
            J(i,j,:) = [(I(i,j-1)+I(i,j+1))/2, I(i,j), (I(i-1,j)+I(i+1,j))/2];
        elseif mod(i,2) == 0 % Red pixel
            J(i,j,:) = [(I(i-1,j)+I(i+1,j))/2, (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, (I(i,j-1)+I(i,j+1))/2];
        else % Blue pixel
            J(i,j,:) = [I(i,j), (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, (I(i,j-1)+I(i,j+1))/2];
        end
    end
end
end

function J = internal_nearest_gbrg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if mod(i+j,2) == 0 % Green pixel
            % J(i,j,:) =
        elseif mod(i,2) == 0 % Red pixel
            % J(i,j,:) =
        else % Blue pixel
            % J(i,j,:) =
        end
    end
end
end

function J = internal_bilinear_grbg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if mod(i+j,2) == 0 % Green pixel
            J(i,j,:) = [(I(i,j-1)+I(i,j+1))/2, I(i,j), (I(i-1,j)+I(i+1,j))/2];
        elseif mod(j,2) == 0 % Red pixel
            J(i,j,:) = [I(i,j), (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, (I(i,j-1)+I(i,j+1))/2];
        else % Blue pixel
            J(i,j,:) = [(I(i-1,j)+I(i+1,j))/2, (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4, I(i,j)];
        end
    end
end
end

function J = internal_nearest_grbg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
for i = 2:m-1
    for j = 2:n-1
        if mod(i+j,2) == 0 % Green pixel
            % J(i,j,:) =
        elseif mod(j,2) == 0 % Red pixel
            % J(i,j,:) =
        else % Blue pixel
            % J(i,j,:) =
        end
    end
end
end
