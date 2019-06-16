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
	output = open("convert_"+sys.argv[1], "a")
	line = line.replace("x64/Release", "/Users/jiyun/darknet")
#	print (line)
	output.write(line)

input.close()
