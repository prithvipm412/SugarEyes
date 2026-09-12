function check_parity(n_samples)
%CHECK_PARITY Compare this MATLAB/Octave preprocessing port (preprocess.m)
% against the real Python-cached outputs it's supposed to reproduce
% (data/cache/<dataset>/<file>.png, produced by scripts/build_cache.py from
% the matching data/raw/<dataset>/images/<file>.png).
%
% VERIFIED under GNU Octave 11.3.0 -- see matlab/README.md for the measured
% numbers and why this gate uses the 99th percentile of the per-pixel
% difference rather than the max: the max is dominated by a thin ring of
% FOV-boundary pixels where OpenCV's contour tracing and MATLAB/Octave's
% bwboundaries pick a very slightly different circle (sub-pixel, but
% enough to shift a background/retina edge by a pixel or two, which is a
% ~150-level jump at that one boundary). That's a real, expected property
% of comparing two independently-implemented circle-fitting algorithms, not
% a bug -- the p99 statistic characterises the other >99% of pixels, which
% is what actually matters for whether downstream grading agrees.
%
% Run from the repository root: matlab/check_parity.m must find
% data/cache/manifest.csv relative to the current directory, the same
% convention every Python script in this repo uses (see AGENTS.md: no
% hardcoded paths).

if nargin < 1
    n_samples = 20;
end

addpath('matlab');
manifest_path = 'data/cache/manifest.csv';
if ~isfile(manifest_path)
    error('check_parity:noManifest', '%s not found -- run scripts/build_cache.py first', manifest_path);
end

rows = read_manifest_csv(manifest_path);
rng(42);  % fixed seed, matching this project's "fix seeds everywhere" rule
idx = randperm(numel(rows), min(n_samples, numel(rows)));

all_diffs = [];
n_checked = 0;
for i = idx
    row = rows(i);
    raw_path = fullfile('data', 'raw', row.dataset, 'images', row.filename);
    cache_path = row.path;
    if ~isfile(raw_path) || ~isfile(cache_path)
        continue  % data/raw is gitignored -- may not be present on every machine
    end

    raw = imread(raw_path);
    py_cached = imread(cache_path);
    [resized, ~] = preprocess(raw, size(py_cached, 1));

    if ~isequal(size(resized), size(py_cached))
        error('check_parity:shapeMismatch', '%s: MATLAB shape %s != Python shape %s', ...
            cache_path, mat2str(size(resized)), mat2str(size(py_cached)));
    end

    d = abs(double(resized) - double(py_cached));
    all_diffs = [all_diffs; d(:)]; %#ok<AGROW>
    n_checked = n_checked + 1;
end

if n_checked == 0
    error('check_parity:noData', 'no raw+cached image pairs found on disk to compare');
end

p99 = prctile(all_diffs, 99);
mean_diff = mean(all_diffs);
max_diff = max(all_diffs);
frac_gt20 = mean(all_diffs > 20) * 100;

printf('checked %d images\n', n_checked);
printf('mean abs diff:  %.3f / 255\n', mean_diff);
printf('p99 abs diff:   %.1f / 255\n', p99);
printf('max abs diff:   %.1f / 255 (expected to be boundary-ring pixels, see header comment)\n', max_diff);
printf('pixels with diff > 20: %.2f%%\n', frac_gt20);

TOLERANCE_P99 = 25;  % set from the measured ~6-17 range on 8 real APTOS
% images plus margin -- not a value handed down from a spec, since the
% two crop algorithms are only expected to closely agree, not match bit
% for bit. Tighten this once measured on a larger sample.
assert(p99 <= TOLERANCE_P99, 'p99 abs diff %.1f exceeds tolerance %.1f', p99, TOLERANCE_P99);
printf('PHASE 6 PREPROCESSING PARITY: PASS (p99 %.1f <= %.1f)\n', p99, TOLERANCE_P99);
end


function rows = read_manifest_csv(path)
% Minimal CSV reader (path,dataset,split,grade,has_masks) avoiding
% readtable, which real MATLAB has but Octave does not -- this way the
% same function runs unmodified in both.
fid = fopen(path, 'r');
header = fgetl(fid); %#ok<NASGU>
rows = struct('path', {}, 'dataset', {}, 'filename', {});
while true
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    parts = strsplit(line, ',');
    path_str = parts{1};
    [~, name, ext] = fileparts(path_str);
    rows(end + 1) = struct( ... %#ok<AGROW>
        'path', path_str, ...
        'dataset', parts{2}, ...
        'filename', [name ext]);
end
fclose(fid);
end
