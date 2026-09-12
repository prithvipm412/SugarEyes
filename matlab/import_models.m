function import_models()
%IMPORT_MODELS Import the ONNX grading model into MATLAB and save it as a
% .mat so later scripts (run_pipeline.m, evaluate.m) don't re-import on
% every run. Uses importNetworkFromONNX (Deep Learning Toolbox, R2023b+).
%
% NOT VERIFIED -- there is no MATLAB installation in the environment this
% was written in (see matlab/README.md for why, and what to do instead).
% importNetworkFromONNX has no equivalent in GNU Octave, so unlike
% preprocess.m this could not be executed against the real
% models/grading.onnx before being committed. What follows is a best-effort
% port written by reading models/grading.onnx's actual export code
% (src/drscreen/grading/model.py's export_onnx: a timm resnet34 backbone
% with num_classes=0, feeding a 4-unit Linear head -- standard Conv/
% BatchNorm/ReLU/Add/GlobalAveragePool/Gemm ops only, opset 13, no custom
% PyTorch ops), which is why placeholder layers are not expected -- but
% "not expected" is a prediction, not a measurement.

onnx_path = fullfile('models', 'grading.onnx');
if ~isfile(onnx_path)
    error('import_models:missing', '%s not found', onnx_path);
end

net = importNetworkFromONNX(onnx_path, InputDataFormats = "BCSS");
% "BCSS" (batch, channel, spatial, spatial) matches the ONNX export's
% input shape [batch, 3, 224, 224] -- see export_onnx's dummy input.

placeholder_idx = find_placeholder_layers(net);
if ~isempty(placeholder_idx)
    names = {net.Layers(placeholder_idx).Name};
    error('import_models:placeholderLayers', ...
        ['importNetworkFromONNX produced %d placeholder layer(s): %s. ' ...
         'Per AGENTS.md''s Phase 6 fallback, rebuild resnet34 natively with ' ...
         'imagePretrainedNetwork and fine-tune the head instead of debugging ' ...
         'this import further.'], numel(placeholder_idx), strjoin(names, ', '));
end
fprintf('import_models: no placeholder layers, %d layers total\n', numel(net.Layers));

feature_layer = find_last_conv_layer(net);
fprintf('import_models: Grad-CAM feature layer guess: %s\n', feature_layer);
fprintf('import_models: VERIFY this manually (analyzeNetwork(net) or inspect net.Layers)\n');
fprintf('import_models: before trusting run_pipeline.m''s gradCAM call.\n');

save(fullfile('models', 'grading_matlab.mat'), 'net', 'feature_layer');
fprintf('import_models: saved models/grading_matlab.mat\n');
end


function idx = find_placeholder_layers(net)
idx = [];
for i = 1:numel(net.Layers)
    if contains(class(net.Layers(i)), 'PlaceholderLayer')
        idx(end + 1) = i; %#ok<AGROW>
    end
end
end


function layer_name = find_last_conv_layer(net)
% Grad-CAM needs the name of the last conv block (Python's model.py:
% target_layer() returns backbone.layer4, resnet34's last residual stage).
% importNetworkFromONNX does not preserve that name, so this searches the
% imported layer graph structurally instead of hardcoding a guess: the
% last Convolution/Add/ReLU-ish layer before the network's global pooling
% or first fully-connected layer.
names = {net.Layers.Name};
classes = cellfun(@class, num2cell(net.Layers), 'UniformOutput', false);

head_idx = find(contains(classes, 'FullyConnected') | contains(classes, 'GlobalAveragePooling'), 1, 'first');
if isempty(head_idx)
    error('import_models:noHead', ...
        'could not find a FullyConnected or GlobalAveragePooling layer to anchor the search -- inspect net.Layers manually');
end

conv_like = find(contains(classes(1:head_idx - 1), 'Convolution') | ...
                 contains(classes(1:head_idx - 1), 'Add') | ...
                 contains(classes(1:head_idx - 1), 'Relu'), 1, 'last');
if isempty(conv_like)
    error('import_models:noConvLayer', 'no conv-like layer found before %s -- inspect net.Layers manually', names{head_idx});
end
layer_name = names{conv_like};
end
