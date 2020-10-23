function [ meanMetric, allMetrics, frames] = evaluationFunc( options, metrics, idx)
%EVALUATIONFUNC Evaluate result with the metric
%   result: array of cells containing the predicted saliency map
%   data: the ground truth data
%   metricName: the name of metric
%       -"similarity": Similarity
%       -"CC": CC
%       -"AUC_Borji": AUC_Borji
%       -"AUC_Judd": AUC_Judd
%       -"AUC_shuffled": sAUC
%   if the ground truth cannot be found, e.g. testing data, the central
%   gaussian will be taken as ground truth automatically.

% GlobalParameters;
%SALICONGlobalParameters;
%addpath(METRIC_DIR);
% assert(length(result)==length(data));
%availableMetric = {'similarity','CC', 'AUC_Judd', 'AUC_Borji', 'AUC_shuffled'};
% assert(any(strcmp(metricName, availableMetric)));
% if strcmp(metricName, 'AUC_shuffled')
%     try
%         load(TRAIN_DATA_PATH);
%     catch
%         fprintf('Training data missing!\n');
%     end
% end

postfix = '.png';

% fh = str2func(metricName);

%%
% frames = dir(fullfile([options.IMG_DIR '*' postfix]));
% disp(fullfile([options.IMG_DIR '*' postfix]));

% nframe = length(frames);
    
%% we evaluate at most 50000 randomly selected frames in one dataset for efficiency
% if nframe>5
%    k = randperm(nframe);
%    frames = frames(k(1:5));
% end

for x=601:700
    temp = dir(fullfile(['/ssd_scratch/cvit/samyak/DHF1K/val/' strcat('0',int2str(x)) '/images/' '*' postfix]));
%     disp(fullfile(fullfile(['/ssd_scratch/cvit/samyak/DHF1K/val/' strcat('0',int2str(x)) '/images/' '*' postfix])));
    temp = temp(idx:32:length(temp))
    if x==601
        frames = temp;
    else
        frames = cat(1, frames, temp);
    end
end
allMetrics = zeros(length(frames),length(metrics));

for i = 1:length(frames)
    gt_fold = frames(i).folder;
    gt_fold = strrep(gt_fold, '\','/');
    gt_name = frames(i).name;
%     continue;
    map_gt_path = strrep(gt_fold,'/images', '/maps/');
    fix_gt_path = strrep(gt_fold,'/images', '/fixation/maps/');
    map_eval_path = strrep(gt_fold, options.DS_GT_DIR, options.SALIENCY_DIR);
    % disp(map_gt_path)
    saliency_path = [map_gt_path, gt_name];
    
%     if ~exist(saliency_path,'file')
%         continue;
%     end
    
    fixation_path = [fix_gt_path, strrep(gt_name, postfix, '.mat')];
    
    load(fixation_path);
    
    result = double(imread([map_eval_path(1:end-6), gt_name]));
    % disp([map_eval_path(1:end-6), gt_name]);
    result = result(:,:,1);
    result = imresize(result, [size(I,1) size(I,2)]);
    for j=1:length(metrics)
        metricName = metrics{j};
        
        fh = str2func(metricName);
        if any(strcmp(metricName, {'similarity','CC', 'EMD'}))
            if exist(saliency_path, 'file')
                I = double(imread(saliency_path))/255;
                
                allMetrics(i,j) = fh( result, I);
            else       
                allMetrics(i,j) = nan;
            end
        elseif any(strcmp(metricName, {'AUC_Judd', 'AUC_Borji','NSS'}))
            if exist(fixation_path, 'file')
                load(fixation_path);
                I = double(I);
                allMetrics(i,j) = fh( result, I);
            else       
                allMetrics(i,j) = nan;
            end       
        elseif strcmp(metricName, 'AUC_shuffled')
            if exist(fixation_path, 'file')
                load(fixation_path);
                I = I>0;
                I = double(I);
                ids = randsample(length(frames), min(10,length(frames)));
                fixation_point = zeros(0,2);
                for k = 1:min(10,length(frames))
                    fx_name = frames(ids(k)).name;
                    fx_fold = frames(ids(k)).folder;
                    fx_fold = strrep(fx_fold, '\','/');

                    fix_path = strrep(fx_fold,'/images', '/fixation/maps/');
                    fixation_pathx = [fix_path, strrep(fx_name, postfix, '.mat')];
                    
                    Ix = load(fixation_pathx);
                    Ix = double(Ix.I);
                    training_resolution = size(Ix);
                    rescale = size(result)./training_resolution;
                    [fx, fy]= find(Ix);
                    pts = vertcat([fy fx]);
                    fixation_point = [fixation_point; pts.*repmat(rescale, size(pts,1), 1)];
                end
                otherMap = makeFixationMap(size(result), fixation_point);
                allMetrics(i,j) = fh( result, I, otherMap);
            else       
                allMetrics(i,j) = nan;
            end 
        else
            allMetrics(i,j) = nan;
        end
    end
end
meanMetric = zeros(length(metrics), 1);
for j=1:length(metrics)
    cnt = 0;
    sm = 0;
    for i = 1:length(frames)
        if isnan(allMetrics(i,j))==0
            sm = sm + allMetrics(i,j);
            cnt = cnt+1;
        end
        % allMetrics(:,j)(isnan(allMetrics(:,j))) = [];
    end
    meanMetric(j) = sm/cnt;
end
end



