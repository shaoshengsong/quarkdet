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
        
    def load_data_mosaic(self,idx):
        """采用固定的中心点即4张图片，均分大小
        
        """

        imgs, annots ,w_scales,h_scales = [], [], [], []
        indices = [idx] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        per_image_size=160
        height =320
        width = 320


        index_list = []
        for i, index in enumerate(indices):
            #print("index:",index)    
            index_list.append(index)
            img = self.load_image(index)
            imgs.append(img)
            annot = self.load_annotations(index)
            annots.append(annot)
            
            h, w=img.shape[:2]
            
            w_scale = per_image_size /w
            h_scale =  per_image_size /h
            w_scales.append(w_scale)
            h_scales.append(h_scale)

        w = 160 
        h = 160 
        imgs[0] = cv2.resize(imgs[0],(w, h))
        imgs[1] = cv2.resize(imgs[1],(w, h))
        imgs[2] = cv2.resize(imgs[2],(w, h))
        imgs[3] = cv2.resize(imgs[3],(w, h))

        result_image = np.zeros((height, width, 3))
        
        
        result_image[0:h,0:w] = imgs[0]
        result_image[0:h,w: w * 2] = imgs[1]
        result_image[h:h * 2 ,0:w] = imgs[2]   
        result_image[h:h * 2, w:w * 2  ] = imgs[3]
        
        annots[0][:,0] *= w_scales[0]
        annots[0][:,1] *= h_scales[0]
        annots[0][:,2] *= w_scales[0]
        annots[0][:,3] *= h_scales[0]
        
        # 第二张图像的坐标的变化  右上
        annots[1][:,0] *= w_scales[1]
        annots[1][:,1] *= h_scales[1]
        annots[1][:,2] *= w_scales[1]
        annots[1][:,3] *= h_scales[1]
        
        annots[1][:,0] += w  #坐标向右移动            
        annots[1][:,2] += w

        # 第三张图像的坐标的变化  左下
        annots[2][:,0] *= w_scales[2]
        annots[2][:,1] *= h_scales[2]
        annots[2][:,2] *= w_scales[2]
        annots[2][:,3] *= h_scales[2]
        
        annots[2][:,1] += h  #坐标向下移动            
        annots[2][:,3] += h
        # 第四张图像的坐标的变化  左下
        annots[3][:,0] *= w_scales[3]
        annots[3][:,1] *= h_scales[3]
        annots[3][:,2] *= w_scales[3]
        annots[3][:,3] *= h_scales[3]
        
        annots[3][:,1] += h  #坐标向下移动            
        annots[3][:,3] += h
        annots[3][:,0] += w  #坐标向右移动            
        annots[3][:,2] += w
        

        result_annot = np.concatenate((annots[0], annots[1], annots[2], annots[3]), axis=0)
        # print("orginal result_annot:",np.rint(result_annot))
        img_info = self.get_per_img_info(idx)
        # #------------------------------------------------
        t_w = result_annot[:, 2] - result_annot[:, 0]
        t_h = result_annot[:, 3] - result_annot[:, 1]
        area = t_w * t_h
        result_annot = result_annot[np.where(area>=self.mosaic_area)]
        
        # print("change result_annot:",np.rint(result_annot))
        # #------------------------------------------------
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
        # if 1:
        #     meta = dict(img=img,
        #     img_info=img_info,
        #     gt_bboxes=ann['bboxes'],
        #     gt_labels=ann['labels']) 
        #     print("original meta:",meta) 
            
            
        if whether < self.mosaic_probability and self.load_mosaic:
            meta = self.load_data_mosaic(idx)
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
    
    
    
    
    
    
    
#  [ 95.  47. 113. 115.   1.]
#  [ 24.   1.  76. 150.   1.]
#  [  9.   3.  30. 117.   1.]
#  [ 82.  58.  87.  77.   1.]
#  [ 91.  53.  97.  72.   1.]
#  [144.  50. 149.  64.   1.]
#  [123.  56. 126.  70.   1.]
#  [105.  54. 108.  61.   1.]
#  [ 61.  25.  82. 153.  41.]
#  [ 99.  80. 120.  91.  41.]
#  [ 11.  59.  16.  71.   1.]
#  [ 90.  55.  92.  65.   1.]
#  [ 53.  89.  62.  99.  41.]
#  [ 86.  55.  88.  60.   1.]
#  [129.  53. 133.  56.   3.]
#  [126.  53. 129.  57.   3.]
#  [ 82.  75.  85.  77.  41.]
#  [135.  53. 137.  60.   1.]
#  [277.  90. 286. 115.  43.]
#  [118. 251. 122. 253.  34.]
#  [201. 213. 259. 286.  19.]
    
# gt box 大小 将被过滤掉的框如下
# [105.  54. 108.  61.   1.]
# [ 86.  55.  88.  60.   1.]
# [129.  53. 133.  56.   3.]
# [126.  53. 129.  57.   3.]
# [ 82.  75.  85.  77.  41.]
# [135.  53. 137.  60.   1.]
# [118. 251. 122. 253.  34.]