function evaluate()
%EVALUATE Reproduce Phase 2's headline grading numbers in MATLAB: ROC/AUC
% (perfcurve), bootstrap CIs on sensitivity/specificity (bootci), a
% confusion matrix (confusionchart), and quadratic weighted kappa.
%
% NOT VERIFIED: depends on run_pipeline.m's grading step, which itself is
% unverified (see that file's header and matlab/README.md). The metric
% *definitions* below are transcribed to match src/drscreen/eval/metrics.py
% exactly (same threshold direction, same confidence level), but the
% numbers this produces have not been checked against Python's -- that is
% the actual point of this file once it can be run, not something already
% confirmed here.
%
% Threshold and expected Python numbers both come from models/manifest.json
% (committed, real, from the actual Kaggle-trained model -- see README.md):
% referable_threshold=0.395, val sensitivity=0.9058, val specificity=0.9480,
% val QWK=0.8744.

addpath('matlab');
manifest = jsondecode(fileread(fullfile('models', 'manifest.json')));
threshold = manifest.referable_threshold;

rows = read_manifest_csv(fullfile('data', 'cache', 'aptos_val_manifest.csv'));
n = numel(rows);
true_grades = zeros(n, 1);
pred_grades = zeros(n, 1);
referable_scores = zeros(n, 1);

mat_path = fullfile('models', 'grading_matlab.mat');
if ~isfile(mat_path)
    error('evaluate:noModel', '%s missing -- run import_models.m first', mat_path);
end
loaded = load(mat_path, 'net');
net = loaded.net;

for i = 1:n
    raw = imread(rows(i).path);  % already 448x448 cache -- preprocess.m's
    % crop+resize is idempotent on a square image already inscribed with
    % its own FOV mask, so re-running it here is safe, matching how
    % pipeline.py's grade_and_extract_features also takes pre-cached images.
    [img448, ~] = preprocess(raw, 448);
    img224 = imresize(img448, [224 224]);
    normalized = (single(img224) / 255.0 - 0.5) / 0.5;
    logits = squeeze(extractdata(predict(net, dlarray(normalized, 'SSCB'))))';
    cum_probs = cumprod(1 ./ (1 + exp(-logits)));
    pred_grades(i) = sum(cum_probs > 0.5);
    referable_scores(i) = cum_probs(2);
    true_grades(i) = rows(i).grade;
end

is_referable = true_grades >= 2;

[x, y, ~, auc] = perfcurve(is_referable, referable_scores, true);
figure; plot(x, y); xlabel('FPR'); ylabel('TPR'); title(sprintf('ROC (AUC=%.4f)', auc));

sens_fn = @(t, s) sum((s >= threshold) & t) / max(sum(t), 1);
spec_fn = @(t, s) sum((s < threshold) & ~t) / max(sum(~t), 1);
sensitivity = sens_fn(is_referable, referable_scores);
specificity = spec_fn(is_referable, referable_scores);
% 'percentile' explicitly -- bootci's default is BCa, which is a different
% (better-calibrated, but different) method from the plain percentile
% bootstrap eval/metrics.py's bootstrap_ci uses; forcing 'percentile'
% keeps this comparable to the Python number instead of silently comparing
% two different statistical procedures.
sens_ci = bootci(2000, {sens_fn, is_referable, referable_scores}, 'type', 'percentile', 'alpha', 0.05);
spec_ci = bootci(2000, {spec_fn, is_referable, referable_scores}, 'type', 'percentile', 'alpha', 0.05);

fprintf('sensitivity: %.4f [%.4f, %.4f] (Python val: 0.9058)\n', sensitivity, sens_ci(1), sens_ci(2));
fprintf('specificity: %.4f [%.4f, %.4f] (Python val: 0.9480)\n', specificity, spec_ci(1), spec_ci(2));
fprintf('AUC: %.4f\n', auc);

qwk = quadratic_weighted_kappa(true_grades, pred_grades, 5);
fprintf('QWK: %.4f (Python val: 0.8744)\n', qwk);

figure; confusionchart(true_grades, pred_grades);
end


function kappa = quadratic_weighted_kappa(y_true, y_pred, n_classes)
O = zeros(n_classes);
for i = 1:numel(y_true)
    O(y_true(i) + 1, y_pred(i) + 1) = O(y_true(i) + 1, y_pred(i) + 1) + 1;
end
[i_idx, j_idx] = meshgrid(0:n_classes - 1, 0:n_classes - 1);
W = ((i_idx - j_idx) .^ 2) / (n_classes - 1) ^ 2;
row_marg = sum(O, 2);
col_marg = sum(O, 1);
E = (row_marg * col_marg) / sum(O(:));
kappa = 1 - sum(W(:) .* O(:)) / sum(W(:) .* E(:));
end


function rows = read_manifest_csv(path)
fid = fopen(path, 'r');
fgetl(fid);  % header
rows = struct('path', {}, 'grade', {});
while true
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    parts = strsplit(line, ',');
    rows(end + 1) = struct('path', parts{1}, 'grade', str2double(parts{4})); %#ok<AGROW>
end
fclose(fid);
end
