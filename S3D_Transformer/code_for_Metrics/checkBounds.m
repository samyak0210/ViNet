function data = checkBounds(dim,data)
    pts = round(data);
    valid = sum((pts(:,1:2) <= repmat([dim(2) dim(1)],[size(pts,1) 1])),2); % pts < image dimensions
    valid = valid + sum((pts(:,1:2) > 0),2); % pts > 0
    data = data(find(valid==4),:);
end