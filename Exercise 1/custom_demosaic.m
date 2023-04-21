function RGB = custom_demosaic(input_image, bayertype, method)
% This function is merely a selector for the appropriate demosaic function,
% for performance reasons. Using this method we avoid any branches in the
% demosaic logic, providing a notable speed up.
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
alpha = 4/8;
beta = 5/8;
gamma = 6/8;

for i = 3:m-2
    for j = 3:n-2
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Blue pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            red_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [red_pixel, green_pixel, I(i,j)];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) && mod(i, 2) == 1 % Green pixel in red row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) && mod(i, 2) == 0 % Green pixel in blue row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        else % Red pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            blue_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [I(i,j), green_pixel, blue_pixel];
        end
    end
end
end

function J = internal_bilinear_bggr_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
alpha = 4/8;
beta = 5/8;
gamma = 6/8;

for i = 3:m-2
    for j = 3:n-2
        if (mod(i,2) == 0 && mod(j, 2) == 0) % Red pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            blue_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [I(i,j), green_pixel, blue_pixel];
        elseif ((mod(i,2) == 0 || mod(j, 2) == 0) && mod(i,2) == 0) % Green pixel in red row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif ((mod(i,2) == 0 || mod(j, 2) == 0) && mod(i,2) == 1) % Green pixel in blue row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        else % Blue pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            red_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [red_pixel, green_pixel, I(i,j)];
        end
    end
end
end

function J = internal_bilinear_gbrg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
alpha = 4/8;
beta = 5/8;
gamma = 6/8;

for i = 3:m-2
    for j = 3:n-2
        if (mod(i+j,2) == 0 && mod(i,2 == 0)) % Green pixel in red row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif (mod(i+j,2) == 0 && mod(i,2 == 1)) % Green pixel in blue row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif mod(i,2) == 0 % Red pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            blue_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [I(i,j), green_pixel, blue_pixel];
        else % Blue pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            red_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [red_pixel, green_pixel, I(i,j)];
        end
    end
end
end

function J = internal_bilinear_grbg_demosaic(I)
[m,n] = size(I);
J = zeros(m,n,3);
alpha = 4/8;
beta = 5/8;
gamma = 6/8;

for i = 3:m-2
    for j = 3:n-2
        if (mod(i+j,2) == 0 && mod(i,2 == 0)) % Green pixel in blue row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i-1,j) + I(i+1,j)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i-2,j) + I(i+2,j)) + 1/10 * I(i,j-2) + 1/10 * I(i,j+2));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif (mod(i+j,2) == 0 && mod(i,2 == 0)) % Green pixel in red row
            red_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            blue_pixel = beta * (I(i,j) + 4/5 * (I(i,j-1) + I(i,j+1)) - 1/5 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1) + I(i,j-2) + I(i,j+2)) + 1/10 * I(i-2,j) + 1/10 * I(i+2,j));
            J(i,j,:) = [red_pixel, I(i,j), blue_pixel];
        elseif mod(j,2) == 0 % Red pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            blue_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [I(i,j), green_pixel, blue_pixel];
        else % Blue pixel
            green_pixel = (I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1))/4 + alpha * (I(i,j) - 1/4 * (I(i,j-2)+I(i,j+2)+I(i-2,j)+I(i+2,j)));
            red_pixel = gamma * (I(i,j) + 1/3 * (I(i-1,j-1) + I(i-1,j+1) + I(i+1,j+1) + I(i+1,j-1)) - 1/4 * (I(i-2,j) + I(i+2,j) + I(i,j-2) + I(i,j+2)));
            J(i,j,:) = [red_pixel, green_pixel, I(i,j)];
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
            J(i,j,:) = [I(i-1, j-1), I(i-1,j), I(i,j)];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            J(i,j,:) = [I(i, j-1), I(i,j), I(i-1,j)];
        else % Red pixel
            J(i,j,:) = [I(i,j), I(i, j-1), I(i-1, j-1)];
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
            J(i,j,:) = [I(i,j), I(i, j-1), I(i-1, j-1)];
        elseif (mod(i,2) == 0 || mod(j, 2) == 0) % Green pixel
            J(i,j,:) = [I(i-1,j), I(i, j), I(i, j-1)];
        else % Blue pixel
            J(i,j,:) = [I(i-1,j-1), I(i-1, j), I(i, j)];
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
            J(i,j,:) = [I(i-1,j), I(i, j), I(i, j-1)];
        elseif mod(i,2) == 0 % Red pixel
            J(i,j,:) = [I(i,j), I(i-1, j), I(i-1, j-1)];
        else % Blue pixel
            J(i,j,:) = [I(i-1,j-1), I(i-1, j), I(i, j)];
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
            J(i,j,:) = [I(i,j-1), I(i,j), I(i-1,j)];
        elseif mod(j,2) == 0 % Red pixel
            J(i,j,:) = [I(i,j), I(i-1,j), I(i-1,j-1)];
        else % Blue pixel
            J(i,j,:) = [I(i-1,j-1), I(i-1,j), I(i,j)];
        end
    end
end
end
