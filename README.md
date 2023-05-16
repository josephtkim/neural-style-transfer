# neural-style-transfer

## Neural Style Transfer Overview

The paper "A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576) describes an approach for style transfer that utilizes the learned feature maps of a CNN. It takes the neural representations of the feature maps to separate and then combine the content and style of images. The key finding in the paper is that the representations for content and style in a CNN are separable.

Given two images, one as the Content (denoted $p$) and the other image as the Style (denoted $a$), Neural Style Transfer will combine the two so that the result image ($x$) conserves the content of $p$ and the style of $a$.

With Neural Style Transfer, we utilize a pre-trained network on an image classification task, such as the VGG19 model trained on ImageNet, which has learned features through many training images. VGG19 works well for Neural Style Transfer because it was trained to be able to generalize on many different types of images, by defining the most important features of the images that are independent from other noise. Layer of the CNN model serve as feature extractors.

Instead of updating the weights of this pre-trained network, we only update the pixels of the generated image, so there is no need to train the CNN model further.

VGG19 architecture:  
![image](./VGG19.png)

For Neural Style Transfer, the last 3 fully connected layers of VGG19 are removed, as they aren't necessary. We only care about the feature maps from the convolution layers.

As an example, these images might represent the features learned through 3 successive layers of a CNN:  
![image](./hierarchical_features.png)

The first layer learns very simple edges, lines, etc. The second layer learns more advanced features such as eyes, noses, ears, and the final layer learns features for entire faces. In NST, the features of the later layers would have the "content" we want to preserve in the generated image.

#### Content
We can obtain the content for an $image$ by feeding an image through the CNN, and sampling activations in the later layers of the VGG19 network. The output sample $C(image)$ gives the content of the input.

#### Style
We can obtain the style for an $image$ by feeding the image through the same CNN, and sampling the activations from several layers of the CNN. A Gram matrix is computed from these activations, which gives the style $S(image)$.

The goal of NST is to update the generated image $x$ to have the content of $p$ and the style of $a$ through gradient descent, so that $C(x) = C(p)$ and $S(x) = S(a)$. Through training, we update the image and minimize losses, described below.

## Loss functions
There are two loss functions, one for the content loss and one for the style loss. The aim of NST is to minimize the total of these two losses. Optionally, another loss function for total variation loss can be added too, which can help with visual coherence.

#### Content Loss
Let $P^l$ and $F^l$ be the respective feature representations in layer $l$ of the content image $p$ and generated image $x$. The content loss is then the squared-error loss between them:

$𝓛_{content}(p, x, l) = \frac{1}{2} \sum_{i,j}(F_{ij}^l - P_{ij}^l)^2$

$F_{ij}^l$ is the activation of the $i^{th}$ filter at position $j$ in layer $l$.

This loss aims to minimize the differences of activations of higher layers to preserve the content.

#### Style Loss
For the style loss, we first calculate the feature correlations by Gram matrix $G^l$, where $G_{ij}^l$ is the inner product of the vectorized feature map $i$ and $j$ in layer $l$:

$G_{ij}^l = ∑_{k}F_{ik}^l F_{jk}^l$

The style loss is the mean-squared distance between the entries of the Gram matrix for the original image $a$ and the generated image $x$, represented as $A^l$ and $G^l$ respectively, for each layer $l$.

The layer's contribution to the total loss is:  
$E_l = \frac{1}{4 N_l^2 M_l^2} \sum_{i,j}(G_{ij}^l - A_{ij}^l)^2$

where $N_l$ is the number of distinct filters in the layer, and $M_l$ is the size of each filter map (height times the width of the feature map).

and the total loss is:  
$𝓛_{style}(a, x) = \sum_{l} w_l E_l$

$w_l$ are weighting factors of the contribution of each layer to the total loss. In the paper, they use $w_l$ = 1/5 for the layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', and 'conv5_1', and $w_l$ for all the other layers. It is equal to one divided by the number of active layers with a non-zero loss-weight $w_l$. 

#### Total Loss
The loss function that NST minimizes is the total of the content and style losses, with weighting factors $α$ and $β$.
$𝓛_{total}(p, a, x) = α𝓛_{content}(p, x) + β𝓛_{style}(a, x)$

During training, the pixel values for the generated image will be updated based on the gradient of the total loss function.

## Results
The following settings typically resulted in decent outputs:
- Content layers: 'block4_conv1'
- Style layers: 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'
- `alpha = 1e1`
- `beta = 1e6`
- `total_variation_weight = 5e-2`
- learning rate = 1
- 500 epochs
- Adam optimizer with `epsilon = 1e-1`

### Example outputs
<img src="./1-1.png" width="480" alt="Paris" />

<img src="./2-2.png" width="480" alt="Dog" />

<img src="./3-1.png" width="480" alt="Roof Piece" />

<img src="./4-2.png" width="480" alt="Grass Field" />

<img src="./5-1.png" width="480" alt="Moonlight" />

<img src="./6-4.png" width="480" alt="Steins Gate" />

<img src="./7-1.png" width="480" alt="Howl's Moving Castle" />
