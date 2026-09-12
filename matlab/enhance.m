function out = enhance(img, mask, method, varargin)
%ENHANCE CLAHE, illumination normalization, or denoising, restricted to
% pixels inside mask. Port of src/drscreen/preprocess/enhance.py. Named to
% match this file (enhance.m) for the same reason preprocess.m is -- see
% that file's header comment.
%
% NOT independently verified under Octave: Octave's image package (2.20.1,
% the version installed while building this) does not implement
% adapthisteq, so the 'clahe' path below could not be executed here. The
% others (imgaussfilt, bilateral filtering) use functions Octave does
% provide, but were not run against real data the way preprocess.m was --
% see matlab/README.md. All three are also not currently called from
% run_pipeline.m's live inference path, mirroring pipeline.py: this project's
% actual quality-gated inference never applies CLAHE/illumination-norm/
% denoise today, only preprocess_image's crop+resize (see pipeline.py's
% ScreeningPipeline.run). Ported anyway because AGENTS.md's Phase 6 plan
% asks for it explicitly.
%
%   out = enhance(img, mask, 'clahe', clipLimit)         % default clipLimit 2.0
%   out = enhance(img, mask, 'illumination', sigmaFrac)  % default sigmaFrac 0.05
%   out = enhance(img, mask, 'denoise')

switch method
    case 'clahe'
        clip_limit = 2.0;
        if ~isempty(varargin)
            clip_limit = varargin{1};
        end
        out = clahe_green(img, mask, clip_limit);
    case 'illumination'
        sigma_frac = 0.05;
        if ~isempty(varargin)
            sigma_frac = varargin{1};
        end
        out = illumination_normalize(img, mask, sigma_frac);
    case 'denoise'
        out = denoise_bilateral(img, mask);
    otherwise
        error('enhance:unknownMethod', 'unknown method: %s', method);
end
end


function out = clahe_green(img, mask, clip_limit)
% CLAHE on the green channel only, matching enhance.py's default mode="green".
out = img;
channel = img(:, :, 2);
enhanced = adapthisteq(channel, 'ClipLimit', clip_limit / 100, 'NumTiles', [8 8]);
% clipLimit/100: MATLAB's adapthisteq ClipLimit is normalized to [0, 1]
% (fraction of the tile histogram), unlike OpenCV's createCLAHE, whose
% clipLimit is an unnormalized histogram-bin count. This scaling is an
% approximation, not a derived equivalence -- unverified, no adapthisteq
% in the Octave environment this was written in. Re-tune once run for real.
out(:, :, 2) = uint8(double(enhanced) .* mask + double(channel) .* ~mask);
end


function out = illumination_normalize(img, mask, sigma_frac)
% Ben Graham style: subtract a heavy Gaussian blur, rescale to uint8.
[h, w, ~] = size(img);
sigma = max(1.0, sigma_frac * max(h, w));
blurred = imgaussfilt(img, sigma);
diff = double(img) - double(blurred);
normalized = (diff - min(diff(:))) / (max(diff(:)) - min(diff(:)) + 1e-8) * 255.0;
out = img;
for c = 1:size(img, 3)
    out(:, :, c) = uint8(normalized(:, :, c) .* mask + double(img(:, :, c)) .* ~mask);
end
end


function out = denoise_bilateral(img, mask)
% Bilateral filter, matching enhance.py's default method="bilateral".
% imbilatfilt ships in MATLAB's Image Processing Toolbox (R2018b+);
% Octave's image package does not have it, so this path is entirely
% unverified.
denoised = imbilatfilt(img, 2500, 5);  % degreeOfSmoothing, spatialSigma --
% chosen to approximate cv2.bilateralFilter(d=5, sigmaColor=50, sigmaSpace=50),
% not derived from an exact parameter mapping between the two libraries.
out = img;
for c = 1:size(img, 3)
    out(:, :, c) = uint8(double(denoised(:, :, c)) .* mask + double(img(:, :, c)) .* ~mask);
end
end
