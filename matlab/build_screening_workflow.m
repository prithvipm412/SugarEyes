function build_screening_workflow()
%BUILD_SCREENING_WORKFLOW Programmatically construct screening_workflow.slx,
% a SimEvents mirror of src/drscreen/sim/screening_des.py's district
% screening queueing model (see that file's own header comment: it names
% this script as its Phase 6 Simulink counterpart).
%
% BY FAR THE LEAST VERIFIED FILE IN matlab/. Not just "not run" like the
% rest of Phase 6 -- the exact SimEvents block library path strings and
% port-connection details below are written from general knowledge of
% SimEvents' block set, not confirmed against any specific MATLAB release,
% because there is no MATLAB installation anywhere in this environment to
% check them against (see matlab/README.md). Block library paths have
% changed across MATLAB versions before. If this errors on `add_block`
% (most likely failure mode), building the model by hand in the Simulink
% canvas from matlab/README.md's parameter table is the recommended path,
% not debugging this script -- it will be faster and won't leave you
% guessing whether a fix here is correct.
%
% Requires the SimEvents add-on. If unavailable, AGENTS.md's stated
% fallback is Stateflow + plain Simulink blocks -- not attempted here,
% to avoid compounding one unverifiable approach with a second.

model = 'screening_workflow';
if bdIsLoaded(model)
    close_system(model, 0);
end
new_system(model);
open_system(model);

% Parameters mirror SimulationConfig's defaults exactly (100,000
% patients/year, 250 working days x 8h -> 50/hour; see
% screening_des.py:annual_volume_to_hourly_rate).
arrivals_per_hour = 50;
n_nurses = 40;      % n_centres(20) x nurses_per_centre(2)
capture_mean_s = 180;
upload_s = (5.0 * 8) / 5.0;   % image_size_mb x 8 / bandwidth_mbps
inference_s = 1.1;
n_reviewers = 5;
review_mean_s = 180;
auto_clear_threshold = 0.95;

add_block('simevents/Generators/Time-Based Entity Generator', [model '/Arrivals'], ...
    'GenerateEntityAfterEachEvent', 'off', 'EntityGenerationDistribution', 'Exponential', ...
    'MeanGenerationInterval', num2str(3600 / arrivals_per_hour));

add_block('simevents/Attributes/Set Attribute', [model '/Assign confidence'], ...
    'EntityType', 'Patient', 'Attributes', 'confidence_grade0');
% VERIFY: real data should be assigned via a bootstrap-resampled lookup
% against data/cache/fusion_dataset.npz (exported to a .mat by
% import_models.m-style code), not a placeholder distribution -- wiring
% that lookup is not attempted here; see screening_des.py's
% _load_grade_confidence_pairs for what this must reproduce.

add_block('simevents/Queues/FIFO Queue', [model '/Nurse queue'], []);
add_block('simevents/Servers/N-Server', [model '/Capture'], ...
    'Capacity', num2str(n_nurses), 'ServiceTimeDistribution', 'Exponential', ...
    'ServiceTimeMean', num2str(capture_mean_s));

add_block('simevents/Servers/Single Server', [model '/Upload'], ...
    'ServiceTimeDistribution', 'Constant', 'ServiceTime', num2str(upload_s));
add_block('simevents/Servers/Single Server', [model '/AI inference'], ...
    'ServiceTimeDistribution', 'Constant', 'ServiceTime', num2str(inference_s));

add_block('simevents/Gates and Switches/Entity Output Switch', [model '/Auto-clear switch'], ...
    'Priority', 'Attribute name', 'PortAttribute', 'confidence_grade0', ...
    'SwitchCriteria', 'Threshold', 'Threshold', num2str(auto_clear_threshold));
% VERIFY: whether "Entity Output Switch" exposes a direct attribute
% threshold like this, or whether the threshold test needs an explicit
% upstream "Attribute Function"/MATLAB Function block producing a 0/1
% routing attribute, depends on the SimEvents version -- this is the
% single most likely line in this file to need hand-fixing.

add_block('simevents/Queues/FIFO Queue', [model '/Reviewer queue'], []);
add_block('simevents/Servers/N-Server', [model '/Review'], ...
    'Capacity', num2str(n_reviewers), 'ServiceTimeDistribution', 'Exponential', ...
    'ServiceTimeMean', num2str(review_mean_s));

add_block('simevents/Sinks/Entity Sink', [model '/Auto-cleared out'], []);
add_block('simevents/Sinks/Entity Sink', [model '/Reviewed out'], []);

% Wiring: Arrivals -> Assign confidence -> Nurse queue -> Capture ->
% Upload -> AI inference -> Auto-clear switch -> {Auto-cleared out,
% Reviewer queue -> Review -> Reviewed out}. Left unconnected here --
% add_line's exact port-name arguments for SimEvents blocks (which differ
% from ordinary Simulink signal ports) are the second most likely thing
% to need version-specific fixing, and doing it in the canvas by dragging
% is far more reliable than guessing port names blind. See
% matlab/README.md for the full block list and parameter table.

save_system(model, fullfile('matlab', [model '.slx']));
fprintf('build_screening_workflow: blocks placed, WIRING NOT COMPLETED -- open %s.slx and connect manually, see file header\n', model);
end
