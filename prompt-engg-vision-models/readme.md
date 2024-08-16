# L1 Overview
- Basic of Prompt Engg.
Prompt can be - Text, image, video , audio

Other References :
- Sharon Zhou - How diffusion model works


# L2 Image Segmentation 
Assigning labels to each pixel
- SAM (Segment Anything Model by Meta)
- FastSAM (50 times faster than SAM using 32x32 image size)

### Type of Image Segmentations :
1. Semantic Image Segmentation
2. Instance Image Segmentation
3. Panoptic Image Segmentation

![alt text](sam_prompt.png)
![alt text](sam.png)
![alt text](fastSam.png)

### MobileSAM

![alt text](mobileSAM.png)


# L3 : Object Detection
ViT : Vision Transformers

**Text prompt --> TO -> Bounding Box**

Vision Transformer for Open-World Localization

![alt text](owl-vit.png)

# L4 : Image Generation

- Using diffusion models
- In-painting

![alt text](diffusionModel.png)


**Hyper Parameter**
Quality id influenced by -
1. No of Inferences : Higher the value more realistic and high resolution
2. Guidance Value 
3. Strength : unique to image to imgae - how much noise to add and remove. May not need to tweak - need to be tested.
4. Negative prompt : Image to not look like, e.g. "cartoon"

![alt text](guidance.png)

