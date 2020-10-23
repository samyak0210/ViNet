function [shufMap] = createShuffmap1(eyeMap_all)
%% Shuf Map
frameMax = length(eyeMap_all);
idx = 1:frameMax;
shufMap = zeros(size(eyeMap_all{1}.eyeMap));
for zidx=idx
    shufMap = shufMap + eyeMap_all{zidx}.eyeMap;
end
shufMap = (shufMap>0);
