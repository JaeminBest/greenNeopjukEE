# greenNeopjukEE
================
KAIST traffic light control system using video image processing based on DNN and adversarial NN


Description
-----------
KAIST safety-team still use human-power for handling traffic and crosswalk. Therefore, we suggest deep-learning based traffic handling system.
We used YOLONET for object(human,car,ducks) detection and adversarial neural etwork for traffic handling. We also compare this adversarial NN with our handmade algorithm.


Requirement
-----------
- python 3.5.2 environment
- anaconda
- yoloNet (download link : https://pjreddie.com/darknet/yolo/)
- yoloNet python library (download link : https://github.com/qqwweee/keras-yolo3)
- requried pip packages :
.. code-block:: text
  tensorflow         1.6.0
  keras              2.1.5
  tensorflow-gpu     1.0.1
  other packages needed for above


Start
-----
1. download yoloNet
2. construct tensorflow conda environment 
.. code-block:: text
  conda create -n deep python=3.5.2 tensorflow=1.6.0 tensorflow-gpu=1.0.1 keras=2.1.5
3. download yoloNet python library
4. 




Implementation in detail
------------------------
- /run.py 

run all process in one command

- /server

server, for information share and decision making for traffic handling


- /test

test dataset for traffic handling demo





Developer Guide
---------------
- localhost:xxxx/
- localhost:xxxx/admin
- localhost:xxxx/admin/show_all_user
- localhost:xxxx/admin/show_one_user
- localhost:xxxx/admin/test_register
- localhost:xxxx/admin/test_unregister
- localhost:xxxx/admin/show_one_image
- localhost:xxxx/admin/display_one_image
