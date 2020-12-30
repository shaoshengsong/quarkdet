import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from .base import BaseDataset
from quarkdet.util import distance2bbox, bbox2distance, overlay_bbox_cv
import random
import math
from quarkdet.util import cfg
_COLORS = np.array([
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.50, 0.5, 0
            ]).astype(np.float32).reshape(-1, 3)

class CocoDataset(BaseDataset):

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
    
    def show(self,meta, class_names):

            all_box = meta['gt_bboxes']
    
            img = meta['img'].astype(np.float32) / 255
            for i, box in enumerate(all_box):
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                color=(0, 255, 0) 
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            return img
        
    def load_mosaic_data_single_scale(self,idx):

                imgs, annots = [], [] 
                indices = [idx] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
       
                index_list = []
                for i, index in enumerate(indices):
                    #print("index:",index)    
                    index_list.append(index)
                    img = self.load_image(index)
                    imgs.append(img)
                    annot = self.load_annotations(index)
                    annots.append(annot)



    
                scale = 0.5 
                height = 320
                width = 320
                w = width * scale
                h = height * scale
                imgs[0] = cv2.resize(imgs[0],(int(w), int(h)))   
                imgs[1] = cv2.resize(imgs[1],(int(w), int(h)))
                imgs[2] = cv2.resize(imgs[2],(int(w), int(h)))
                imgs[3] = cv2.resize(imgs[3],(int(w), int(h)))
         
                


                result_image = np.zeros((height, width, 3))
                result_image[0:int(h),0:int(w)] = imgs[0]
                result_image[0:int(h),int(w):(int(w * 2))] = imgs[1]
                
                result_image[int(h):(int(h * 2) ),int(w):(int(w * 2 ) )] = imgs[2]
                result_image[int(h):(int(h * 2) ),0:int(w)] = imgs[3]

                annots[0][:, :4] *= scale
                annots[1][:, :4] *= scale
                annots[2][:, :4] *= scale
                annots[3][:, :4] *= scale

                annots[1][:, 0] += int(w)
                annots[1][:, 2] += int(w)

                annots[2][:, 0] += int(w)
                annots[2][:, 2] += int(w)
                annots[2][:, 1] += int(h)
                annots[2][:, 3] += int(h)

                annots[3][:, 1] += int(h)
                annots[3][:, 3] += int(h)

                result_annot = np.concatenate(
                    (annots[0], annots[1], annots[2], annots[3]), axis=0)

                img_info = self.get_per_img_info(idx)
                meta = dict(img=result_image,
                            img_info=img_info,
                            gt_bboxes=result_annot[:, 0:4].astype(np.float32),
                            gt_labels=result_annot[:, 4].astype(np.int64))
                return meta
        
        
     
 
    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
   
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
       
        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """

        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)

        
        #print("img_ids:",len(self.img_ids))
        whether=random.random() 
        # print("whether:",whether)  
        # print("idx:",idx)  
        #把两个meta的数据都打印出来 对比
        
        
        # if 1:                
        #     meta1 = self.load_mosaic_data_single_scale(idx) 
        #     img1 = meta1['img'].astype(np.float32) / 255
        #     print("change meta:",meta1)
            
        #     #meta['img'] = img
        #     # cv2.imshow('whether:', img)  
        #     # cv2.waitKey(0) 
        #     #print("change meta:",meta)
        # if 1:
        #     meta = dict(img=img,
        #     img_info=img_info,
        #     gt_bboxes=ann['bboxes'],
        #     gt_labels=ann['labels']) 
        #     print("original meta:",meta) 
            
            
        if False: #whether< 0.5:
            #meta = self.load_mosaic_data_single_scale(idx)    
            #meta = self.load_mosaic_data_single_scaleA(idx)
            meta = self.load_mosaic_data(idx)
            # img1 = meta['img'].astype(np.float32) / 255
            # cv2.imshow('whether:', img1)  
            # cv2.waitKey(0) 
            #print("change meta:",meta)
        else:
            meta = dict(img=img,
            img_info=img_info,
            gt_bboxes=ann['bboxes'],
            gt_labels=ann['labels'])               
            #print("original meta:",meta)  opencv H x W xC
            
        # img_test=self.show(meta, cfg.class_names)
        # cv2.imshow('img_test:', img_test)
        # cv2.waitKey(0) 

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)) #H x W x C 转化为 C x H x W
        return meta

    # def get_val_data(self, idx):
    #     """
    #     Currently no difference from get_train_data.
    #     Not support TTA(testing time augmentation) yet.
    #     :param idx:
    #     :return:
    #     """
    #     # TODO: support TTA
    #     return self.get_train_data(idx)
    
    
    
    def get_val_data(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)
        
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta
    #增加可视化的工作
    
    # def show_result(self, img, dets, class_names, score_thres=0, show=True, save_path=None):
    #         result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
    #         cv2.imshow('det', result)
            
     
    def load_image(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        

        return img #.astype(np.float32)

    def load_annotations(self, idx):

        
        
        annotations_ids = self.coco_api.getAnnIds(
            imgIds=self.img_ids[idx], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco_api.loadAnns(annotations_ids)
        for _, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            if a['bbox'][2] > 0 and a['bbox'][3] > 0:
                annotation[0, :4] = a['bbox']
                annotation[0, 4] = a['category_id'] 

                annotations = np.append(annotations, annotation, axis=0)

        # transform from [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max] # bbox = [x1, y1, x1 + w, y1 + h]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
 
        return annotations        
        
    def load_mosaic_data_single_scaleA(self,idx):
    

                imgs, annots = [], []
                img = self.load_image(idx)
                imgs.append(img)
                annot = self.load_annotations(idx)
                annots.append(annot)

                index_list, index = [idx], idx
                for _ in range(3):
                    while index in index_list:
                        index = np.random.randint(0, len(self.img_ids))
                        
                    print("index:",index)    
                    index_list.append(index)
                    img = self.load_image(index)
                    imgs.append(img)
                    annot = self.load_annotations(index)
                    annots.append(annot)

                # 第1，2，3，4张图片按顺时针方向排列，1为左上角图片，先计算出第2张图片的scale，然后推算出其他图片的最大resize尺寸，为了不让四张图片中某几张图片太小造成模型学习困难，scale限制为在0.25到0.75之间生成的随机浮点数。
                # 图像采用等比例u缩小
                scale1 = 0.5 #np.random.uniform(0.2, 0.8)
                height1=320
                width1=320
                height2=320
                width2=320
                height3=320
                width3=320
                height4=320
                width4=320
                #height1, width1, _ = imgs[0].shape

                imgs[0] = cv2.resize(imgs[0],(int(width1 * scale1), int(height1 * scale1)))

                max_height2, max_width2 = int(height1 * scale1), width1 - int(width1 * scale1)
                
                #height2, width2, _ = imgs[1].shape
                scale2 = max_height2 / height2
                
                if int(scale2 * width2) > max_width2:
                    scale2 = max_width2 / width2
                    
                imgs[1] = cv2.resize(imgs[1],(int(width2 * scale2), int(height2 * scale2)))

                max_height3, max_width3 = height1 - int(
                    height1 * scale1), width1 - int(width1 * scale1)
                #height3, width3, _ = imgs[2].shape
                scale3 = max_height3 / height3
                if int(scale3 * width3) > max_width3:
                    scale3 = max_width3 / width3
                    
                imgs[2] = cv2.resize(imgs[2],(int(width3 * scale3), int(height3 * scale3)))

                max_height4, max_width4 = height1 - int(height1 * scale1), int(
                    width1 * scale1)
                #height4, width4, _ = imgs[3].shape
                scale4 = max_height4 / height4
                if int(scale4 * width4) > max_width4:
                    scale4 = max_width4 / width4
                imgs[3] = cv2.resize(imgs[3],(int(width4 * scale4), int(height4 * scale4)))

                # 最后图片大小和原图一样
                height1=320
                width1=320
                final_image = np.zeros((height1, width1, 3))
                final_image[0:int(height1 * scale1),
                            0:int(width1 * scale1)] = imgs[0]
                final_image[0:int(height2 * scale2),int(width1 * scale1):(int(width1 * scale1) +int(width2 * scale2))] = imgs[1]
                
                final_image[int(height1 * scale1):(int(height1 * scale1) +int(height3 * scale3)),int(width1 * scale1):(int(width1 * scale1) +int(width3 * scale3))] = imgs[2]
                final_image[int(height1 * scale1):(int(height1 * scale1) +
                                                int(height4 * scale4)),
                            0:int(width4 * scale4)] = imgs[3]

                annots[0][:, :4] *= scale1
                annots[1][:, :4] *= scale2
                annots[2][:, :4] *= scale3
                annots[3][:, :4] *= scale4

                annots[1][:, 0] += int(width1 * scale1)
                annots[1][:, 2] += int(width1 * scale1)

                annots[2][:, 0] += int(width1 * scale1)
                annots[2][:, 2] += int(width1 * scale1)
                annots[2][:, 1] += int(height1 * scale1)
                annots[2][:, 3] += int(height1 * scale1)

                annots[3][:, 1] += int(height1 * scale1)
                annots[3][:, 3] += int(height1 * scale1)

                final_annot = np.concatenate(
                    (annots[0], annots[1], annots[2], annots[3]), axis=0)

                #sample = {'img': final_image, 'annot': final_annot, 'scale': 1.}
                
                # print("final_annots:",final_annots)
                # cv2.imshow('combined_img', combined_img)  
                # cv2.waitKey(0) 
                img_info = self.get_per_img_info(idx)
                meta = dict(img=final_image,
                img_info=img_info,
                # gt_bboxes=final_annots[:, 1:].astype(np.float32),
                # gt_labels=final_annots[:, 4].astype(np.int64))  annots[:, :4]
                gt_bboxes=final_annot[:, 0:4].astype(np.float32),
                gt_labels=final_annot[:, 4].astype(np.int64))
                return meta    

    def merge_anno(self, idx):
        ann = self.get_img_annotation(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels'].reshape(-1, 1)
        # normal_boxes = self.xyxy2ccwh(gt_bboxes, h, w)
        labels = np.hstack((gt_labels, gt_bboxes))
        return labels

    def load_imageA(self, idx, img_size=320):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        h0, w0 = img.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        
    def load_mosaic_data(self, idx):
        count_sample=len(self.img_ids)
        img_size=320
        labels4 = []
        s = img_size
        mosaic_border = [-img_size // 2, -img_size // 2]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
        indices = [idx] + [random.randint(0, count_sample - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_imageA(index, img_size)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = self.merge_anno(index)
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                # labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                # labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                # labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                # labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
                labels[:, 1] = x[:, 1] + padw
                labels[:, 2] = x[:, 2] + padh
                labels[:, 3] = x[:, 3] + padw
                labels[:, 4] = x[:, 4] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate
        # Augment
        # img4, labels4, warp_matrix = self.random_perspective(img4, labels4,
        #                                         degrees=self.hyp['degrees'],
        #                                         translate=self.hyp['translate'],
        #                                         scale=self.hyp['scale'],
        #                                         shear=self.hyp['shear'],
        #                                         perspective=self.hyp['perspective'],
        #                                         border=mosaic_border)  # border to remove
        # return img4, labels4
        meta = dict(img=img4,
                    img_info=self.data_info[idx],
                    gt_bboxes=labels4[:, 1:].astype(np.float32),
                    gt_labels=labels4[:, 1].astype(np.int64))
                    #warp_matrix=warp_matrix)
        return meta
    
    
    
    
    
    
    
    
    
    
