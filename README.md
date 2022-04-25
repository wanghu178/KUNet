# TODO

- [ ] Submit source code
- [x] Submit implementation details
- [ ] Submit appendex
- [ ] Perfect link to the paper
- [ ] Citation
- [ ] Acknowledgment
- [ ] Getting Started

# KUNet[Paper Link]

IJCAI2022 

# Implementation details

For the Head, Tail, D block of KIB module and KIC module, we use $3\times 3$ convolution with a step size of 1, and for the X and Y branches in the KIB module, we use $1 \times 1$ convolution. The activation function of all networks is ReLU function. Except for input and output layers, the channels of feature maps are all 64. The down-sampling operation uses a $3 \times 3$ convolution operation with a step size of 2. Up-sampling operation uses PixelShuffle[1]. We use ADAM optimizer[2] with the learning rate of $2e^{-4}$ decayed by a factor of 2 after every $20K$ iterations. The batch size is 12. All models are built on the PyTorch framework and trained with NVIDIA GeForce RTX 2080 SUPER. The total training time is about 6 days. 

Once the final version is accepted, we will upload our source code, model , train and test sets

1. W.Shi,J.Caballero,F.Husźar,J.Totz,A.P.Aitken,R.Bishop,D.Rueckert,andZ.Wang.Real-timesingleimageandvideosuper-resolutionusinganefficientsub-pixelconvolutionalneuralnetwork.InCVPR,pages1874–1883,201

2. H. Scharr. Optimal filters for extended optical flow. In IWCM, pages 14–29. Springer

