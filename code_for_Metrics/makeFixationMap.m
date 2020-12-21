% Draw binary fixation map with size=dim, with ones in the x,y positions
% given in the pts and zeros everywhere else
function map = makeFixationMap(dim,pts)
    pts = round(pts(:,1:2));
    map = zeros(dim);
    pts = checkBounds(dim,pts);
    ind = sub2ind(dim,pts(:,2),pts(:,1));
%     map(ind) = 1;
    map(ind) = map(ind) + 1;
end