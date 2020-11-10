# Image Super Resolution using SRCNN
In SRCNN, the steps are as follows:

Bicubic interpolation is done first to upsample to the desired resolution.\
Then 9×9, 1×1, 5×5 convolutions are performed to improve the image quality.\
For the 1×1 conv, it was claimed to be used for non-linear mapping of the low-resolution (LR) image vector and the high-resolution (HR) image vector.

## The SRCNN Network

In SRCNN, actually the network is not deep. There are only 3 parts, patch extraction and representation, non-linear mapping, and reconstruction as shown in the figure below:\
Image : https://miro.medium.com/max/792/1*RxT4yZtXFkQ47Fe7huHe_w.png/

For more details, refer to: https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c

