% This finds the information-gain of the saliencyMap over a baselineMap

function score = IG(saliencyMap, fixationMap, baseMap)
% saliencyMap is the saliency map
% fixationMap is the human fixation map (binary matrix)
% baselineMap is another saliency map (e.g. all fixations from other images)

%%
%  format, resize, and normalize maps
% map1 = im2double(fixationMap); 
map2 = im2double(imresize(saliencyMap, size(fixationMap)));
% map1 = (map1-min(map1(:)))/(max(map1(:))-min(map1(:)));
map2 = (map2-min(map2(:)))/(max(map2(:))-min(map2(:)));

% compute per-pixel information gain (over baseline, if provided)
% map1 = map1/sum(map1(:));
map2 = map2/sum(map2(:));
locs = logical(fixationMap(:));

if nargin>2 && ~isempty(baseMap)
    mapb = im2double(imresize(baseMap, size(fixationMap)));
    mapb = (mapb-min(mapb(:)))/(max(mapb(:))-min(mapb(:)));
    mapb = mapb/sum(mapb(:));
    % map1_log = log2(eps+map1(locs))-log2(eps+mapb(locs));
    map2_log = log2(eps+map2(locs))-log2(eps+mapb(locs));
    
% if no baseMap is provided, IG calculation will be very similar to KL
else
    % map1_log = log2(eps+map1(locs));
    map2_log = log2(eps+map2(locs));
end

%%
% score = mean(log2(eps+map1(locs))-log2(eps+mapb(locs))); 
score = mean(map2_log); 