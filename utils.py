

## takes a homogeneous list of dictionaries and converts it into a dictionary
def lst_to_dict(lst):
	key_lst = list(lst[0].keys())
	output_dict = {}
	for item in lst:
		for key in key_lst:
			if key in output_dict:
				output_dict[key].append(item[key])
			else:
				output_dict[key] = [item[key]]
	#print (output_dict)
	return output_dict




