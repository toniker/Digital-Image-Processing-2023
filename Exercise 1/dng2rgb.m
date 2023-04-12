function [Csrgb, Clinear, Cxyz, Ccam] = dng2rgb(rawim, XYZ2Cam, wbcoeffs, bayertype , method)
    % White Balance
    % https://rcsumner.net/raw_guide/RAWguide.pdf
    % Section 4.3
    Cxyz = rawim .* wbcoeffs;
    imshow(Cxyz);

    const XYZ2RGB = [[+3.2406, -1.5372, -0.4986], [-0.9689, +1.8758, +0.0415], [+0.0557, -0.2040, +1.0570]];

    Csrgb = Clinear .^ (1/2.2);
end