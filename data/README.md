Description
----------------------
KAIST safety-team still use human-power for handling traffic and crosswalk. Therefore, we suggest deep-learning based traffic handling system.
We used YOLONET for object(human,car,ducks) detection and adversarial neural etwork for traffic handling. We also compare this adversarial NN with our handmade algorithm.

YOLO Marking Tool
-
This program is for windows, but it tracks objects very well.
So, I used this program to labeling and then converting the format to use in YOLONET. (YOLO_text.txt)

https://darkpgmr.tistory.com/m/16

YOLO Training
-
I refer these sites to YOLO training.

https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

https://pgmrlsh.tistory.com/6?category=766787

https://murra.tistory.com/18

For YOLO training, we need train.txt and test.txt, and those file should contain the path of data.
Primarily I got this file from executing yolo marking tool, and to change the path of all data easily, I used "convert_.txt". 


in charge : di-uni
