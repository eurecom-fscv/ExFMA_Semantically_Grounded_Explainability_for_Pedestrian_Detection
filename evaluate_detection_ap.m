% Evaluate detector average precision against P-DESTRE ground truth.
%
% Environment variables:
%   PDESTRE_DETECTION_OUTPUT_DIR
%   PDESTRE_ANNOTATION_DIR

clear;

vis_path = getenv('PDESTRE_DETECTION_OUTPUT_DIR');
if isempty(vis_path)
    vis_path = 'outputs/visible_PDestre_bottom_exp_out';
end

GT_path = getenv('PDESTRE_ANNOTATION_DIR');
if isempty(GT_path)
    GT_path = 'data/P_Destre/annotation';
end

D = dir(fullfile(vis_path, '*_bbox.mat'));
N = size(D, 1);
resize_factor = 1;
resize_factorGT = 4;

annotations = [];
scores = [];
fprintf('Starting...');
for i = 1 : N
    bb_file = fullfile(D(i).folder, D(i).name);
    bboxes = load(bb_file);
    bboxes = bboxes.bboxes;
    if size(bboxes, 1) > 0
        a = bboxes(bboxes(:, 5) > 0.3, 1:4);
        a(:, 3:4) = a(:, 3:4) - a(:, 1:2);
        annotations{i} = a / resize_factor;
        scores{i} = bboxes(bboxes(:, 5) > 0.3, 5);
    else
        annotations{i} = [];
        scores{i} = [];
    end

    annotation_array = [];
    [~, name, ~] = fileparts(D(i).name);
    parts = split(name, '_');
    frame_idx = parts{3};
    fileN = fullfile(GT_path, [parts{1}, '.txt']);
    fileID = fopen(fileN, 'r');
    line = fgetl(fileID);
    if line == -1
        fclose(fileID);
        continue;
    end
    while ischar(line)
        info = split(line, ',');
        frame = info{1};
        if strcmp(frame, frame_idx)
            coor_double = [str2double(info{3}), str2double(info{4}), str2double(info{5}), str2double(info{6})];
            coor_double(coor_double < 0) = 1;
            coor_double = coor_double / resize_factorGT;
            annotation_array = [annotation_array; coor_double]; %#ok<AGROW>
        end
        line = fgetl(fileID);
    end
    annotationsGT{i} = annotation_array;
    fclose(fileID);
end

visible_table = cell2table([annotations', scores']);
GT_table = cell2table(annotationsGT');

[ap, recall, precision] = evaluateDetectionPrecision(visible_table, GT_table);
fprintf('Average precision = %.2f; Recall = %.2f; Precision = %.2f. \n', ap * 100, mean(recall) * 100, mean(precision) * 100);
