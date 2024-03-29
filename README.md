# Neural-Style-Transfer

1. Abstract
2. Implementation
3. Thesis und Colloquium

# Abstract
Throughout history, art regularly had to reinvent itself. Now, machine learning detaches paintings from the canvas, making art take a further step. Previous research was able to apply representations of artistic styles to images. In this work, we present an approach to combine multiple styles for neural style transfer, which we perform by applying adaptive instance normalization (AdaIN). For this, we use a generative adversarial network (GAN) with a few-shot image generator with a content conditioned style encoder (COCO-FUNIT). Using t-SNE, we show that our model can reproduce real-world relations between styles. However, we find that further methods need to be developed to better measure the quality of style intersections.

# Implementation
To run the application, please download the Places365 dataset and add art dataset to folder "data". 

# Thesis und Colloquium
To get an overview on our results please cosinder the [colloquium](https://github.com/fmunzlin/Neural-Style-Transfer/blob/main/Colloquium_munzlinger.pdf). Attached you also find my [thesis](https://github.com/fmunzlin/Neural-Style-Transfer/blob/main/Masterarbeit_Munzlinger.pdf).

# Results

![Heidelberg Brueckenaffe stylized in the style of Picasso](figures/Ape_1_Picasso.jpg)
![Heidelberg Brueckenaffe stylized in the style of Peploe](figures/Ape_2_1_Peploe.jpg)
![Heidelberg Brueckenaffe stylized in the styles of Pollock and Kirchner](figures/ape_2_2_Pollock_kirchner.jpg)
![Heidelberg Brueckenaffe stylized in the styles of Picasso and Manet](figures/Ape_2_2_Picasso_Manet.jpg)
![Heidelberg castle stylized in the styles of Pissarro and Picasso](figures/HD_2_2_pissarro_picasso.jpg)
