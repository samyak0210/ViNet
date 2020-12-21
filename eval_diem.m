function eval_diem(pred_path, annot_path, annot_file, per_frame)

addpath(genpath('./code_for_Metrics'));
 % = [annot_base_path '/annotations/DIEM/'];
%
% pred_path = [pred_path 'diem_test/'];
%
%% read database info
fileID = fopen(annot_file,'r');
txt_data = textscan(fileID,'%s','delimiter','\n'); 
fclose(fileID);
test_data = struct([]);
for ii=1:length(txt_data{1})
    data_split = strsplit(txt_data{1}{ii});
    name = data_split{1};
    nframes = double(length(dir([fullfile(annot_path, name, 'maps') '/*jpg'])))
    % nframes  = str2double(data_split{2});
    test_data(ii).video = name;
    test_data(ii).nframes = nframes;
    test_data(ii).annot_path  = fullfile(annot_path, name);
    test_data(ii).pred_path  = fullfile(pred_path, name);
end

%% evaluate videos

loss_cc = 0;
loss_nss = 0;
loss_sim = 0;
loss_aucj = 0;
loss_sauc = 0;
cnt = 0;
for i=1:length(test_data)
  file_list = dir([test_data(i).pred_path '/*jpg']);
  
  for j=1:test_data(i).nframes
      eyeMap_all{j} = load(fullfile(test_data(i).annot_path, sprintf('fixMap_%05d.mat',j)));
  end
  disp([test_data(i).pred_path '/*jpg']);

  shufMap_all = createShuffmap1(eyeMap_all);
  frame_cc = zeros(length(file_list), 1); 
  frame_sim = zeros(length(file_list), 1); 
  frame_nss = zeros(length(file_list), 1); 
  frame_aucj = zeros(length(file_list), 1); 
  frame_sauc = zeros(length(file_list), 1); 
  fprintf('video %d of %d\n', i, length(test_data));
  for j = 1:length(file_list)
     tmp = strsplit(file_list(j).name,'.');
     tmp = strsplit(tmp{1},'_');
     frame_num = str2double(tmp{end});
     frame_name = fullfile([test_data(i).pred_path], '', file_list(j).name);
     if (frame_num<=test_data(i).nframes)

         I_pred = im2double(imread(frame_name));
         I_pred = I_pred(:,:,1);
         I_eye_name = fullfile(test_data(i).annot_path, 'maps', sprintf('eyeMap_%05d.jpg',frame_num));
         I_eye = im2double(imread(I_eye_name));
         I_bin = eyeMap_all{frame_num};
         % I_pred_post = postprocessMap(I_pred,2);
         %
         salMap = imresize(I_pred,size(I_eye));
         eyeMap = double(I_bin.eyeMap);
         salMap_gt = I_eye;
         shufMap1 = double(shufMap_all);
         shufMap1(eyeMap==1) = 0;
         %% Compute Metrics
         frame_cc(j) = CC(salMap, salMap_gt);
         frame_sim(j) = similarity(salMap, salMap_gt);
         frame_nss(j) = NSS(salMap, eyeMap);
         frame_aucj(j) = AUC_Judd(salMap, eyeMap);
         frame_sauc(j) = AUC_shuffled(salMap, eyeMap, shufMap1);
         if isnan(frame_cc(j))
          disp(frame_cc(j));
         end
     else
     	disp(frame_num);
     end
  end
  %% Compute Averages per video
  disp(length(frame_cc));
  frame_cc(isnan(frame_cc)) = [];
  frame_sim(isnan(frame_sim)) = [];
  frame_nss(isnan(frame_nss)) = [];
  frame_aucj(isnan(frame_aucj)) = [];
  frame_sauc(isnan(frame_sauc)) = [];
  disp("Metric Value")
  fprintf('%.4f \n', mean(frame_cc));
  fprintf('%.4f \n', mean(frame_sim));
  fprintf('%.4f \n', mean(frame_nss));
  fprintf('%.4f \n', mean(frame_aucj));
  fprintf('%.4f \n', mean(frame_sauc));
  %
  disp(length(frame_cc));

  if strcmp(per_frame, 'false')
    disp(per_frame);
	  loss_cc = loss_cc + mean(frame_cc);
	  loss_sim = loss_sim + mean(frame_sim);
	  loss_nss = loss_nss + mean(frame_nss);
	  loss_aucj = loss_aucj + mean(frame_aucj);
	  loss_sauc = loss_sauc + mean(frame_sauc);
	  cnt = cnt + 1;
  else
  	  loss_cc = loss_cc + sum(frame_cc);
  	  loss_sim = loss_sim + sum(frame_sim);
  	  loss_nss = loss_nss + sum(frame_nss);
  	  loss_aucj = loss_aucj + sum(frame_aucj);
  	  loss_sauc = loss_sauc + sum(frame_sauc);
  	  cnt = cnt + length(frame_cc);
  end

end
disp(loss_cc);
disp(cnt);
disp(loss_cc / cnt);
disp(loss_sim / cnt);
disp(loss_nss / cnt);
disp(loss_aucj / cnt);
disp(loss_sauc / cnt);
%%
