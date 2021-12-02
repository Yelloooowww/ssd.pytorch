from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torchvision
import random
import os.path as osp
import matplotlib.pyplot as plt
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from ssd import build_ssd
import pyrealsense2 as rs

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

pipeline.start(config)

def cv2_demo(net):
	def predict(frame):
		image = frame
		# scale each detection back up to the image
		scale = torch.Tensor(image.shape[1::-1]).repeat(2)
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		x = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
		x -= (104.0, 117.0, 123.0)
		x = x.astype(np.float32)
		x = x[:, :, ::-1].copy()
		x = torch.from_numpy(x).permute(2, 0, 1)

		xx = Variable(x.unsqueeze(0))# wrap tensor in Variable
		y = net(xx)
		detections = y.data
		for i in range(detections.size(1)):
			j = 0

			while detections[0, i, j, 0] >= 0.4:
				pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

				labelmap = ["lion", "tiger"]
				cv2.rectangle(frame,
			              (int(pt[0]), int(pt[1])),
			              (int(pt[2]), int(pt[3])),
			              COLORS[i % 3], 5)
				labelmap = ["lion", "tiger"]
				cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
				            FONT, 2, COLORS[i % 3], 2, cv2.LINE_AA)
				print(detections[0, i, j, 0],labelmap[i - 1])
				j += 1
		return frame

	# start video stream thread, allow buffer to fill
	print("[INFO] starting threaded video stream...")
	while True:
		# grab next frame
		key = cv2.waitKey(1) & 0xFF

		frame = pipeline.wait_for_frames()
		color_frame = np.asanyarray(frame.get_color_frame().get_data())
		frame = predict(color_frame)

		# keybindings for display
		if key == ord('p'):  # pause
			while True:
				key2 = cv2.waitKey(1) or 0xff
				cv2.imshow('frame', frame)
				if key2 == ord('p'):  # resume
					break
		cv2.imshow('frame', frame)
		if key == 27:  # exit
			break


if __name__ == '__main__':
	ssd_net = build_ssd('test', 300, 3)
	ssd_net.load_weights('/home/yellow/ssd.pytorch/weight/mydata/mydata_8500.pth')

	cv2_demo(ssd_net.eval())

	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# cleanup
	cv2.destroyAllWindows()
	stream.stop()
