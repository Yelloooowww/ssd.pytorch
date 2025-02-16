"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import random
import json
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import cv2
import numpy as np
if sys.version_info[0] == 2:
	import xml.etree.cElementTree as ET
else:
	import xml.etree.ElementTree as ET

# SUBT_CLASSES = [  # always index 0
#     'missle','backpack','blueline','drill','can']
SUBT_CLASSES = ['lion','tiger']

#SUBT_CLASSES = (  # always index 0
#    'valve', '')

# note: if you used our download scripts, this should be right
SUBT_ROOT = osp.join(HOME, "data/subt_artifact/")


class SUBTAnnotationTransform(object):
	"""Transforms a VOC annotation into a Tensor of bbox coords and label index
	Initilized with a dictionary lookup of classnames to indexes

	Arguments:
		class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
			(default: alphabetic indexing of VOC's 20 classes)
		keep_difficult (bool, optional): keep difficult instances or not
			(default: False)
		height (int): height
		width (int): width
	"""

	def __init__(self, class_to_ind=None, keep_difficult=False):
		self.class_to_ind = class_to_ind or dict(
			zip(SUBT_CLASSES, range(len(SUBT_CLASSES))))
		self.keep_difficult = keep_difficult
	def __call__(self, target, width, height):
		"""
		Arguments:
			target (annotation) : the target annotation to be made usable
				will be an ET.Element
		Returns:
			a list containing lists of bounding boxes  [bbox coords, class name]
		"""
		res = []
		print("target.iter('object')", target.iter('object'))
		for obj in target.iter('object'):
			#difficult = int(obj.find('difficult').text) == 1
			#if not self.keep_difficult and difficult:
			#    continue
			name = obj.find('name').text.lower().strip()
			if name not in self.class_to_ind:
				continue
			bbox = obj.find('bndbox')
			if bbox is not None:
				pts = ['xmin', 'ymin', 'xmax', 'ymax']
				bndbox = []
				for i, pt in enumerate(pts):
					cur_pt = int(bbox.find(pt).text) - 1
					# scale height or width
					cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
					bndbox.append(cur_pt)
				label_idx = self.class_to_ind[name]
				bndbox.append(label_idx)
				res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
				# img_id = target.find('filename').text[:-4]
			else: # For LabelMe tool
				polygons = obj.find('polygon')
				x = []
				y = []
				bndbox = []
				for polygon in polygons.iter('pt'):
					# scale height or width
					x.append(int(polygon.find('x').text) / width)
					y.append(int(polygon.find('y').text) / height)
				bndbox.append(min(x))
				bndbox.append(min(y))
				bndbox.append(max(x))
				bndbox.append(max(y))
				label_idx = self.class_to_ind[name]
				bndbox.append(label_idx)
				res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

		return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class BigCatAnnotationTransform(object):
	"""Transforms a VOC annotation into a Tensor of bbox coords and label index
	Initilized with a dictionary lookup of classnames to indexes

	Arguments:
		class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
			(default: alphabetic indexing of VOC's 20 classes)
		keep_difficult (bool, optional): keep difficult instances or not
			(default: False)
		height (int): height
		width (int): width
	"""

	def __init__(self, class_to_ind=None, keep_difficult=False):
		self.class_to_ind = class_to_ind or dict(
			zip(SUBT_CLASSES, range(len(SUBT_CLASSES))))
		self.keep_difficult = keep_difficult
	def __call__(self, target, width, height):
		"""
		Arguments:
			target (annotation) : the target annotation to be made usable
				will be an ET.Element
		Returns:
			a list containing lists of bounding boxes  [bbox coords, class name]
		"""
		res = []
		xmin = min(target["shapes"][0]['points'][0][1], target["shapes"][0]['points'][1][1])
		ymin = min(target["shapes"][0]['points'][0][0], target["shapes"][0]['points'][1][0])
		xmax = max(target["shapes"][0]['points'][0][1], target["shapes"][0]['points'][1][1])
		ymax= max(target["shapes"][0]['points'][0][0], target["shapes"][0]['points'][1][0])
		if target["shapes"][0]['label']=="lion" : label_ind = 0
		elif target["shapes"][0]['label']=="tiger" : label_ind = 1
		else :
			label_ind = random.random()%2
			print("why else????????????????????")
		res += [ymin/width, xmin/height, ymax/width, xmax/height, label_ind]

		return [res]  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class SUBTDetection(data.Dataset):
	"""VOC Detection Dataset Object

	input is image, target is annotation

	Arguments:
		root (string): filepath to VOCdevkit folder.
		image_set (string): imageset to use (eg. 'train', 'val', 'test')
		transform (callable, optional): transformation to perform on the
			input image
		target_transform (callable, optional): transformation to perform on the
			target `annotation`
			(eg: take in caption string, return tensor of word indices)
		dataset_name (string, optional): which dataset to load
			(default: 'VOC2007')
	"""

	def __init__(self, root,
				 image_sets=['train', 'val'],
				 transform=None, target_transform=BigCatAnnotationTransform(),
				 dataset_name='SUBT'):
		self.root = root
		self.image_set = image_sets
		self.transform = transform
		self.target_transform = target_transform
		self.name = dataset_name
		self._annopath = "/home/yellow/ssd.pytorch/mydata/big_cat/Annotations/"
		self._imgpath = "/home/yellow/ssd.pytorch/mydata/big_cat/JPEGImages/"

		self.ids = list()
		for name in image_sets:
			rootpath = osp.join(self.root)
			for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
				self.ids.append((rootpath, line.strip()))
		print("dataset init done, len=", len(self.ids))


	def __getitem__(self, index):
		im, gt, h, w = self.pull_item(index)

		return im, gt

	def __len__(self):
		return len(self.ids)

	def pull_item(self, index):
		img_id = self.ids[index]

		# image
		original_img = cv2.imread(self._imgpath + img_id[1]+".jpg")
		resized_img = cv2.resize(original_img, dsize=(300, 300))
		height, width, channels = original_img.shape
		# print("height, width, ",height, width)

		# json
		with open(self._annopath + img_id[1] + ".json") as f:
			json_data = json.load(f)
		if self.target_transform is not None:
			target = self.target_transform(json_data, width=width, height=height)
		if self.transform is not None:
			target = np.array(target)
			img, boxes, labels = self.transform(resized_img, target[: , :4], target[: , 4])

			# to rgb
			img = img[:, :, (2, 1, 0)]
			target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
		return torch.from_numpy(img).permute(2, 0, 1), target, height, width

	def pull_image(self, index):
		'''Returns the original image object at index in PIL form

		Note: not using self.__getitem__(), as any transformations passed in
		could mess up this functionality.

		Argument:
			index (int): index of img to show
		Return:
			PIL img
		'''
		img_id = self.ids[index]
		return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

	# def pull_anno(self, index):
	# 	'''Returns the original annotation of image at index
	#
	# 	Note: not using self.__getitem__(), as any transformations passed in
	# 	could mess up this functionality.
	#
	# 	Argument:
	# 		index (int): index of img to get annotation of
	# 	Return:
	# 		list:  [img_id, [(label, bbox coords),...]]
	# 			eg: ('001718', [('dog', (96, 13, 438, 332))])
	# 	'''
	# 	img_id = self.ids[index]
	# 	anno = ET.parse(self._annopath % img_id).getroot()
	# 	gt = self.target_transform(anno, 1, 1)
	# 	return img_id[1], gt

	# def pull_tensor(self, index):
	# 	'''Returns the original image at an index in tensor form
	#
	# 	Note: not using self.__getitem__(), as any transformations passed in
	# 	could mess up this functionality.
	#
	# 	Argument:
	# 		index (int): index of img to show
	# 	Return:
	# 		tensorized version of img, squeezed
	# 	'''
	# 	return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
