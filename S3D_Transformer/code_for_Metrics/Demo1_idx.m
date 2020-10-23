
%% Demo.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
function exitcode = Demo1(model_name, idx)

% METRIC_DIR = 'code_forMetrics';
% addpath(genpath(METRIC_DIR));

%% path to store evaluation results
CACHE = ['cache/'];
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%%
options.Result_path = '/ssd_scratch/cvit/samyak/Results/';
options.DS_path = '/ssd_scratch/cvit/samyak/';

Datasets{1} = 'UCF';
Datasets{2} = 'Hollywood';
Datasets{3} = 'DHF1K';


Metrics{1} = 'CC'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'NSS';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';
% disp(model_name);
exitcode = 0;

Results{1} = model_name;

for i = 3:3 %length(Datasets)
    Datasets{i};
    for k =1: length(Results)
        % saliency prediction results
        options.SALIENCY_DIR = [options.Result_path '/' Results{k} '/'];
        disp(Results{k});
        
        % dataset path
        if isequal(Datasets{i}, 'DHF1K'),
            options.DS_GT_DIR = [options.DS_path Datasets{i} '/val/'];
        else
            options.DS_GT_DIR = [options.DS_path Datasets{i} '/testing/'];
        end
        options.IMG_DIR = [options.DS_GT_DIR, '*/images/'];
        % // disp(options.IMG_DIR);
        % for j = 5:5 %length(Metrics)
            % if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file')                 
            [result, allMetric, ~] = eval1(options, Metrics, idx);
            for j=1:length(result)
                fprintf('%s :%.4f \n', Metrics{j}, result(j));
            end
                % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
                % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'result');
                % std_value = std(allMetric); % calculate std value if you want to
            % else
            %     load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
            % end
            % meanMetric{i}(k,j) = result;
            % fprintf('%s :%.4f \n', Metrics{j}, result);
        % end
    end
end

%%
end