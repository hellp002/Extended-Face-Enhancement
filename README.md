# Extended-Face-Enhancement
extension of final project face enhancement <br>
Member:
- Werapat Wangrungroj

## Project Overview

เนื่องจากว่าใน project เดิม final-project face enhancement ยังไม่สามารถสร้าง model ที่สามารถเข้ากับ goal ของ project ได้มากนัก เพราะว่าตัว model มีขนาด file ที่ใหญ่มาก (~3GB) และมีการใช้ ram ที่สูงมาก (~6GB)
ซึ่ง goal เดิมของเราคืออยากจะใช้ model เพื่อปรับแต่งรูปภาพ แต่เนื่องจากว่าการปรับแต่งรูปภาพมักจะใช้บน mobile device เป็นหลักทำให้จำเป็นต้องสร้าง model ใหม่ที่เป็นแบบ lightweight model เพื่อให้ทำงานบน mobile device ได้
อีกทั้ง project เดิมได้มีการทำ preprocessing data ไม่เหมาะสมทำให้การ train model ใช้เวลานาน(~14 นาที) / 1 epoch แม้จะใช้ GPU ที่ดีๆก็ตาม(A100) ทำให้ใช้ หน่วยประมวลผล บน google colab อย่างมหาศาล จึงไม่สามารถลอง encoder ตัวอื่นๆได้

## Optimization จาก project เดิม

ได้มีการ preprocess ก่อนนำไปเข้า data set เนื่องจากว่า image ของ lapa เดิมมีขนาดที่หลากหลายโดยมีขนาดที่มากกว่า 224x224 แน่ๆ จึงทำการย่อให้เหลือ 224x224 ก่อนและ save image ไว้ใน google colab disk เพื่อลดเวลาการ Load image จาก disk
ทำให้เวลาการ train / 1 epoch เหลือ ~ 2นาที และเปลี่ยนไปใช้ GPU V100 ทำให้ speedup ได้ 7 เท่าและประหยัดหน่วยประมวลผลได้ 2.6 เท่า
และมีการเปลี่ยนการใช้ loss function เป็น focal loss gamma = 2 ทำให้สามารถ segmentation ได้เก่งขึ้นแม้จะใช้ encoder ที่เล็กลง

## Architecture

ได้มีการลองใช้ Architecture แบบ Unet++ และ DeepLabV3+ โดย model4 ใช้เป็นแบบ DeepLabV3+ ส่วน model5 ใช้เป็น Unet++ และวัดผลด้วย mean IoU scoreเพื่อหาว่า model ไหนเหมาะสมกับการนำมาใช้กว่ากันจะเห็นว่าจาก

[model4](https://github.com/hellp002/Extended-Face-Enhancement/blob/main/model_eval_on_LaPa/model4/model4_test_class.csv)

[model5](https://github.com/hellp002/Extended-Face-Enhancement/blob/main/model_eval_on_LaPa/model5/model5_test_class.csv)

จะเห็นว่า performance เฉพาะส่วนที่นำมาใช้ segmentation ของ model 5 ใกล้เคียงกับ model 4 (background,upper lip,mouth,lower lip,hair) และในส่วนการนำไปใช้จริงๆ เหมือนว่า DeepLabV3+ จะดีกว่าเล็กน้อย

[model4](https://github.com/hellp002/Extended-Face-Enhancement/blob/main/model_eval_face_enhancement/model4_result_enhancement.csv)

[model5](https://github.com/hellp002/Extended-Face-Enhancement/blob/main/model_eval_face_enhancement/model5_result_enhancement.csv)

จึงตัดสินใจว่าควรจะใช้ Architecture แบบ DeepLabV3+ และสำหรับการ segmentation ในส่วนใบหน้า และใช้ face detection จาก library dlib 

Loss Plot

![model4_loss](https://github.com/hellp002/Extended-Face-Enhancement/assets/94524977/12d12dc0-6251-42e5-89c8-b8216d5d5487)

![model5_loss](https://github.com/hellp002/Extended-Face-Enhancement/assets/94524977/19876d02-4fd3-4b0f-b782-4f4681296da0)

mIoU Plot

![model4_iou](https://github.com/hellp002/Extended-Face-Enhancement/assets/94524977/08827eb0-0e84-4466-9eac-e3db987f8375)

![model5_iou](https://github.com/hellp002/Extended-Face-Enhancement/assets/94524977/c67439ad-254e-43f8-bbab-c71360719d25)

Encoder | Weights | Params,M
--- | --- | ---
mobilenet_v2 | imagenet | 2M

## Process in Our Method
![](https://github.com/hellp002/Final-Project-Face-Enhancement/assets/94524977/bd943348-f759-46e2-8852-09b0260ee3d3) ![](https://github.com/hellp002/Final-Project-Face-Enhancement/assets/94524977/29ed89a1-7aa0-44d0-bc5d-0652709778e9)
## How to use our method

การ enhancement ใบหน้าเราจำเป็นต้องรัน code ใน [file](eval_model.ipynb) ที่สามารถ run ได้ใน google colab โดยต้อง run code cell จนถึงบรรทัดสุดท้าย และเราจึงจะสามารถเรียกใช้

```python
enhancer(image)
```

ที่รับ input เป็น BGR image โดยมี output เป็น RGB image นอกจากนี้ก่อนเรียก function enhance จำเป็นต้องใช้คำสั่งนี้ครอบก่อน

```python
model.eval()
with torch.no_grad():
 ....
 result = enhancer(image)
```

เพื่อเปลี่ยน model เป็น mode evaluation จะเป็นการเพิ่ม accuracy ให้ model

## Reference for Dataset:

Dataset: [Dataset github](https://github.com/JDAI-CV/lapa-dataset)

Dataset paper: [A New Dataset and Boundary-Attention Semantic Segmentation for Face Parsing](https://aaai.org/ojs/index.php/AAAI/article/view/6832/6686). Yinglu Liu, Hailin Shi, Hao Shen, Yue Si, Xiaobo Wang, Tao Mei. In AAAI, 2020.

## Link and Assets
[train/dev/test set for unet](https://drive.google.com/uc?export=download&id=1XOBoRGSraP50_pS1YPB8_i8Wmw_5L-NG) <br>
[data set that we used](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) <br>
[data set for testing our model](https://drive.google.com/uc?export=download&id=1WeP0mTjUDBt2Zx4JWO0U0xf15jwpsr6V) <br>
[data set for testing (facial mask)](https://drive.google.com/uc?export=download&id=1-sr6XByGYKRdIDuS3MjAWCBnhX1_OnGG) <br>
[data set for testing (face mask)](https://drive.google.com/uc?export=download&id=1K0QTK_GSyai5vNwMgaO3Kh54n5w4Sjtx) <br>
[Model](https://drive.google.com/uc?export=download&id=1_fdYp8trR7mMDWeqjHOhTASp4SQv7RSk) <br>
[dlib model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) <br>
[Medium](https://medium.com/@werapatwangrungroj/face-enhancement-ด้วย-semantic-segmentation-model-และ-facial-landmark-detection-model-2a8c1381b1a8) <br>
[Data Label App](https://imagej.net/ij/download.html) <br>
[Slide Presentation](Face%20Enhancement.pdf) <br>


 
