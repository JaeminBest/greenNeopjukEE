# last update: 2019-05-26 
# by di-uni
# for YOLO, we need to convert the format of output of 'Dark label' 
# the format of Dark label (iname,n[|id,cx,cy,w,h,label])
# ------------------------------------------------------------------------------
# img_name, n
# label_id, x, y, w, h, label_name
# ex)
# frame0001.jpg, 2
# 0, 1562, 554, 139, 93, null
# 3, 1791, 574, 70, 58, null
# ------------------------------------------------------------------------------

import sys

try:
	input = open(sys.argv[1], "r")
except:
	print ('ERROR: check the input file name')
	sys.exit(1)

while True:
	line = input.readline()
	if not line:
		break
	arr = line.split(',')
	output_name = arr[0].replace(".jpg", ".txt")
	output = open("/output "+output_name, "w")
	for i in range (int(arr[1])): 
		line = input.readline()
		arr_data = line.split(',')
		x = float(int(arr_data[1]) / 1920)
		y = float(int(arr_data[2]) / 1080)
		w = float(int(arr_data[3]) / 1920)
		h = float(int(arr_data[4]) / 1080)
		output.write("{} {:0.6f} {:0.6f} {:0.6f} {:0.6f}\n".format(int(arr_data[0]), x, y, w, h))
	
input.close()
