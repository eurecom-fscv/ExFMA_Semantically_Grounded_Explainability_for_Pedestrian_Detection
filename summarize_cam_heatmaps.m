% Average CAM heatmaps over annotated pedestrian boxes.
%
% Environment variables:
%   PDESTRE_CAM_OUTPUT_DIR
%   PDESTRE_ANNOTATION_DIR

clear;

vis_path = getenv('PDESTRE_CAM_OUTPUT_DIR');
if isempty(vis_path)
    vis_path = 'outputs/visible_PDestre_head_exp_out';
end

GT_path = getenv('PDESTRE_ANNOTATION_DIR');
if isempty(GT_path)
    GT_path = 'data/P_Destre/annotation';
end

output_image = 'mean_map_head_out.png';

D = dir(fullfile(vis_path, '*cam.mat'));
N = size(D, 1);
rect_array = [];
for i = 1 : N
    exp_map = load(fullfile(vis_path, D(i).name));
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
            coor_double(coor_double <= 0) = 1;
            coor_double = coor_double / 4;
            coor_int = floor(coor_double);
            if coor_int(3) / coor_int(4) < 0.6
                rect = imcrop(exp_map.grayscale_cam, coor_int);
                rect = imresize(rect, [100, 50]);
                rect_array = cat(3, rect_array, rect);
            end
        end
        line = fgetl(fileID);
    end
    fclose(fileID);
end

avg_rect = mean(rect_array, 3, 'omitnan');
imwrite(avg_rect, output_image);
