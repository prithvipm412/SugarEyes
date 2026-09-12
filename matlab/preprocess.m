function [resized, mask] = preprocess(img, sz)
%PREPROCESS Crop an RGB fundus photo to its retina field of view and
% resize to a fixed square. Port of src/drscreen/preprocess/retina.py's
% preprocess_image (crop_to_retina + resize_cached). Named to match this
% file's name (preprocess.m) -- MATLAB/Octave resolve a call by filename,
% not by the function keyword's name, so a mismatch here silently makes
% the function uncallable under its "real" name (confirmed empirically:
% see matlab/README.md).
%
% VERIFIED under GNU Octave 11.3.0 (image package 2.20.1), not under real
% MATLAB -- see matlab/README.md for exactly what that does and doesn't
% prove. Every function called here (rgb2gray, bwconncomp, bwboundaries,
% imresize with a custom kernel, padarray) is a documented Image Processing
% Toolbox function with the same name and signature in real MATLAB, so this
% is expected to run there unmodified, but that expectation is unconfirmed.
%
%   [resized, mask] = preprocess_image(img, sz)
%   img: HxWx3 uint8 RGB image (NOT BGR -- imread already returns RGB,
%        unlike Python's cv2.imread which needs an explicit BGR2RGB swap)
%   sz:  output side length in pixels (default 448, matching AGENTS.md's
%        cache size)
%   resized: sz x sz x 3 uint8
%   mask:    sz x sz logical, true inside the inscribed FOV circle

if nargin < 2
    sz = 448;
end

[cropped, ~] = crop_to_retina(img);
[resized, mask] = resize_cached(cropped, sz);
end


function [cropped, mask] = crop_to_retina(img)
% Tight square crop around the retina field-of-view circle. Mirrors
% retina.py's crop_to_retina: threshold -> largest connected component ->
% minimum enclosing circle -> square crop, zero-padded if the circle
% extends past the image bounds.
gray = rgb2gray(img);
bw = gray > 10;

cc = bwconncomp(bw, 8);
if cc.NumObjects == 0
    error('preprocess:noFovContour', 'No retina FOV contour found in image');
end
sizes = cellfun(@numel, cc.PixelIdxList);
[~, largest_idx] = max(sizes);
largest_mask = false(size(bw));
largest_mask(cc.PixelIdxList{largest_idx}) = true;

boundaries = bwboundaries(largest_mask, 8, 'noholes');
boundary = boundaries{1};                    % Nx2, columns = [row col] = [y x]
points = [boundary(:, 2), boundary(:, 1)];   % -> [x y], what min_enclosing_circle expects

[center, radius] = min_enclosing_circle(points);
cx = round(center(1));
cy = round(center(2));
r = round(radius);

x0 = cx - r; x1 = cx + r;
y0 = cy - r; y1 = cy + r;
[h, w, ~] = size(img);
pad_left = max(0, -x0);
pad_top = max(0, -y0);
pad_right = max(0, x1 - w);
pad_bottom = max(0, y1 - h);

if any([pad_left, pad_top, pad_right, pad_bottom])
    img = padarray(img, [pad_top, pad_left], 0, 'pre');
    img = padarray(img, [pad_bottom, pad_right], 0, 'post');
    x0 = x0 + pad_left; x1 = x1 + pad_left;
    y0 = y0 + pad_top;  y1 = y1 + pad_top;
end

% Python's img[y0:y1, x0:x1] is 0-indexed with an exclusive upper bound;
% MATLAB/Octave indexing is 1-indexed and inclusive, so the equivalent
% range is (y0+1):y1, (x0+1):x1 -- both give a span of exactly x1-x0 (resp.
% y1-y0) elements.
cropped = img((y0 + 1):y1, (x0 + 1):x1, :);
mask = cached_fov_mask(size(cropped, 1));
end


function mask = cached_fov_mask(sz)
% The FOV mask for a square crop of side sz: the inscribed circle,
% derived analytically rather than by resizing a pixel mask -- see
% retina.py's cached_fov_mask docstring for why (resize interpolation
% only approximates an ideal circle at the boundary).
[xx, yy] = meshgrid(0:(sz - 1), 0:(sz - 1));
center = sz / 2;
mask = (xx - center) .^ 2 + (yy - center) .^ 2 <= center ^ 2;
end


function [resized, mask] = resize_cached(img, sz)
% Resize to (sz, sz). retina.py uses OpenCV's INTER_LANCZOS4 (an 8-tap,
% a=4 Lanczos kernel); MATLAB/Octave's built-in imresize only ships
% 'lanczos2'/'lanczos3', so a custom kernel matching OpenCV's a=4 support
% is passed explicitly via imresize's {kernel, size} form (a standard,
% documented MATLAB imresize feature, not an Octave-only extension).
% Antialiasing is turned off to match OpenCV's resize, which does not
% broaden the kernel on downscale.
resized = imresize(img, [sz sz], {@lanczos4_kernel, 8}, 'Antialiasing', false);
mask = cached_fov_mask(sz);
end


function y = lanczos4_kernel(x)
% Lanczos kernel with a=4, i.e. sinc(x) * sinc(x/4) on [-4, 4]. Written
% out by hand (rather than calling the Signal Processing Toolbox's sinc)
% so this file has no dependency beyond Image Processing Toolbox.
y = mysinc(x) .* mysinc(x / 4);
y(abs(x) >= 4) = 0;
end


function y = mysinc(x)
y = ones(size(x));
nz = x ~= 0;
y(nz) = sin(pi * x(nz)) ./ (pi * x(nz));
end


function [c, r] = min_enclosing_circle(points)
% Welzl's randomized incremental algorithm for the minimum enclosing
% circle of a 2D point set (expected O(n)). points is Nx2 [x y].
% cv2.minEnclosingCircle targets the same well-defined geometric quantity
% via a different (iterative) method -- see matlab/README.md for the
% measured agreement between the two on real fundus images.
idx = randperm(size(points, 1));
points = points(idx, :);
c = points(1, :);
r = 0;
for i = 2:size(points, 1)
    if norm(points(i, :) - c) > r + 1e-7
        [c, r] = circle_with_point(points(1:i, :), points(i, :));
    end
end
end


function [c, r] = circle_with_point(points, p)
c = p;
r = 0;
for i = 1:size(points, 1)
    if norm(points(i, :) - c) > r + 1e-7
        [c, r] = circle_with_two_points(points(1:i, :), p, points(i, :));
    end
end
end


function [c, r] = circle_with_two_points(points, p, q)
c = (p + q) / 2;
r = norm(p - q) / 2;
for i = 1:size(points, 1)
    if norm(points(i, :) - c) > r + 1e-7
        [c, r] = circumcircle(p, q, points(i, :));
    end
end
end


function [c, r] = circumcircle(a, b, c3)
ax = a(1); ay = a(2); bx = b(1); by = b(2); cx = c3(1); cy = c3(2);
d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
if abs(d) < 1e-9
    % Near-collinear: fall back to the circle over the two farthest points.
    dists = [norm(a - b), norm(b - c3), norm(a - c3)];
    [~, worst] = max(dists);
    switch worst
        case 1
            p1 = a; p2 = b;
        case 2
            p1 = b; p2 = c3;
        otherwise
            p1 = a; p2 = c3;
    end
    c = (p1 + p2) / 2;
    r = norm(p1 - p2) / 2;
    return
end
ux = ((ax^2 + ay^2) * (by - cy) + (bx^2 + by^2) * (cy - ay) + (cx^2 + cy^2) * (ay - by)) / d;
uy = ((ax^2 + ay^2) * (cx - bx) + (bx^2 + by^2) * (ax - cx) + (cx^2 + cy^2) * (bx - ax)) / d;
c = [ux, uy];
r = norm(a - c);
end
