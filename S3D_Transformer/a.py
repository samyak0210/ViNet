list_frames = [i for i in range(0,6500)]
len_temporal = 32

# print(list_frames)
arr = []

for i in range(0, len(list_frames) - len(list_frames)%(2*len_temporal-2), 2*len_temporal-2):
    snippet = []
    for j in range(i,i+len_temporal-1):
        if(len(snippet)==0):
            for k in range(j,j+len_temporal):
                snippet.append(list_frames[k])
        else:
            del snippet[0]
            snippet.append(list_frames[j+len_temporal-1])
        # print(snippet, [list_frames[j], list_frames[j+len_temporal-1]])
        arr.extend([list_frames[j], list_frames[j+len_temporal-1]])
 

for i in range(len(list_frames) - len(list_frames)%(2*len_temporal-2), len(list_frames)):
    snippet = []
    for j in range(i - len_temporal+1, i+1):
        snippet.append(list_frames[j])
    # print(snippet, [snippet[0], snippet[len_temporal-1]])
    arr.extend([snippet[0], snippet[len_temporal-1]])

arr = list(set(arr))
assert arr == list_frames

# arr = []
# for i in range(0, len(list_frames), 2*len_temporal-2):
# 	if i+2*len_temporal-2>=len(list_frames):
# 		break

# 	snippet = []
# 	for j in range(i,i+len_temporal-1):
# 		if(len(snippet)==0):
# 			for k in range(j,j+len_temporal):
# 				snippet.append(list_frames[k])
# 		else:
# 			del snippet[0]
# 			snippet.append(list_frames[j+len_temporal-1])

# 		print(snippet, [list_frames[j], list_frames[j+len_temporal-1]])
# 		arr.extend([list_frames[j], list_frames[j+len_temporal-1]])
# tmp_j = j+1
# snippet = []
# for j in range(tmp_j, len(list_frames)):
# 	if(j+len_temporal-1==len(list_frames)):
# 		break
# 	if(len(snippet)==0):
# 		for k in range(j,j+len_temporal):
# 			snippet.append(list_frames[k])
# 	else:
# 		del snippet[0]
# 		snippet.append(list_frames[j+len_temporal-1])

# 	print(snippet, [list_frames[j], list_frames[j+len_temporal-1]])
# 	arr.extend([list_frames[j], list_frames[j+len_temporal-1]])

# arr = set(arr)
# list_frames = [i for i in range(1,64)]
# print(list(arr)==list_frames)