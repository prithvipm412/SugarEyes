function result = run_pipeline(image_path)
%RUN_PIPELINE Load an image, preprocess, grade, Grad-CAM, overlay, display.
% The MATLAB compliance-layer demo script -- "is this MATLAB?" (AGENTS.md
% Phase 6). Mirrors src/drscreen/pipeline.py's grading+Grad-CAM path only
% (not the full quality-gate/lesion/fusion pipeline -- see matlab/README.md
% for why that's a deliberate scope cut, not a missing feature).
%
% NOT VERIFIED end to end: depends on models/grading_matlab.mat from
% import_models.m, which itself could not be run (no MATLAB in this
% environment -- see matlab/README.md). preprocess.m's crop+resize IS
% independently verified (see check_parity.m); the grading/gradCAM half
% below is not.

addpath('matlab');
mat_path = fullfile('models', 'grading_matlab.mat');
if ~isfile(mat_path)
    error('run_pipeline:noModel', '%s missing -- run import_models.m first', mat_path);
end
loaded = load(mat_path, 'net', 'feature_layer');
net = loaded.net;
feature_layer = loaded.feature_layer;

raw = imread(image_path);
[img448, ~] = preprocess(raw, 448);
img224 = imresize(img448, [224 224]);
% Match pipeline.py's _to_tensor: /255, then (x-0.5)/0.5 -> [-1, 1], NOT
% ImageNet mean/std normalization.
normalized = (single(img224) / 255.0 - 0.5) / 0.5;
input = dlarray(normalized, 'SSCB');

logits = predict(net, input);
logits = squeeze(extractdata(logits))';  % 1x4 CORN logits

cum_probs = cumprod(1 ./ (1 + exp(-logits)));
grade = sum(cum_probs > 0.5);
referable_score = cum_probs(2);  % P(grade > 1) == P(grade >= 2)

fprintf('%s: grade=%d referable_score=%.4f\n', image_path, grade, referable_score);

reduction_fcn = @(scores) prod(1 ./ (1 + exp(-scores(1:2))));
% Backprops through the same scalar Python's _ReferableTarget does
% (referable_score = P(grade>=2)), not a class-index softmax score --
% this network has no softmax head. gradCAM's exact name/arg for a custom
% reduction function may not match current MATLAB releases; check
% `help gradCAM` if this line errors -- see matlab/README.md, this is the
% single most likely spot to need a fix.
cam = gradCAM(net, input, reduction_fcn, 'FeatureLayer', feature_layer);

cam_resized = imresize(squeeze(extractdata(cam)), [224 224]);
cam_norm = rescale(cam_resized);
heatmap = ind2rgb(gray2ind(cam_norm, 256), jet(256));
overlay = 0.6 * im2double(img224) + 0.4 * heatmap;

figure;
subplot(1, 3, 1); imshow(img224); title('Preprocessed');
subplot(1, 3, 2); imshow(heatmap); title('Grad-CAM');
subplot(1, 3, 3); imshow(overlay); title(sprintf('Grade %d (p=%.2f)', grade, referable_score));

result = struct('grade', grade, 'referable_score', referable_score, 'overlay', overlay);
end
