% Extract annotated P-DESTRE frames from source videos.
%
% Environment variables:
%   PDESTRE_ANNOTATION_DIR
%   PDESTRE_VIDEO_DIR
%   PDESTRE_FRAME_OUTPUT_DIR

GT_path = getenv('PDESTRE_ANNOTATION_DIR');
if isempty(GT_path)
    GT_path = 'data/P_Destre/annotation';
end

videoPath = getenv('PDESTRE_VIDEO_DIR');
if isempty(videoPath)
    videoPath = 'data/P_Destre/videos';
end

imageDest = getenv('PDESTRE_FRAME_OUTPUT_DIR');
if isempty(imageDest)
    imageDest = 'data/P_Destre/frames';
end

D = dir(fullfile(GT_path, '*.txt'));
N = size(D, 1);
if ~exist(imageDest, 'dir')
    mkdir(imageDest);
end

for i = floor(N / 2) + 1 : N
    dataFile = fullfile(D(i).folder, D(i).name);
    [~, name, ~] = fileparts(dataFile);

    frameIndex = [];
    fileID = fopen(dataFile, 'r');
    line = fgetl(fileID);
    while ischar(line)
        info = split(line, ',');
        frameIndex(end + 1) = str2double(info{1}); %#ok<AGROW>
        line = fgetl(fileID);
    end
    fclose(fileID);
    frameIndex = unique(frameIndex);

    videoFile = fullfile(videoPath, [name, '.MP4']);
    if ~exist(videoFile, 'file')
        videoFile = fullfile(videoPath, [name, '.mp4']);
    end
    v = VideoReader(videoFile);
    for j = 1 : length(frameIndex)
        video = read(v, frameIndex(j));
        imageFile = fullfile(imageDest, sprintf('%s_frame_%d.png', name, frameIndex(j)));
        imwrite(video, imageFile);
    end
end
