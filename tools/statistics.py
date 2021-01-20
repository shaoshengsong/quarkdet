from pycocotools.coco import COCO
# img_path: /media/ubuntu/data/dataset/COCOv1/2017/train2017
# ann_path: /media/ubuntu/data/dataset/COCOv1/2017/annotations/
dataDir='/media/ubuntu/data/dataset/COCOv1/2017/'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds()) # 类别
cat_nms=[cat['name'] for cat in cats] #cat_nms是list类型
#print(type(cat_nms))
#print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
#print(len(cats))

# 错误的方式
# catId = coco.getCatIds(catNms=cat_name)
# 应把cat_name 变成 [cat_name]
# 统计各类的图片数量和GT框数量
for cat_name in cat_nms:
    #print("type(cat_name):",type(cat_name)) #test cat_name是str类型
    catId = coco.getCatIds(catNms=[cat_name])
    #print("type(catId):",type(catId)) #test catId是list所以可以直接传参
    imgId = coco.getImgIds(catIds=catId)
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
    
    
    #下面这段代码是测试，如果输出后面的注释的数字例如[3, 57]表示统计存在错误。
    # 如果输出一个数字表示正确
    #car & carrot
    #ear & teddy bear
    #dog & hot dog
    #-----------------------------------------------------------------------
    # if cat_name == "carrot":
    #     print(catId) #[3, 57]

    # if cat_name == "teddy bear":
    #     print(catId) #[23, 88]

    # if cat_name == "hot dog":
    #     print(catId) #[18, 58]
    #-----------------------------------------------------------------------
                
    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))