# SparseSwin 
---
This repo is the official implementation of "SparseSwin: Swin Transformer with Sparse Transformer Block". <br>

# Introduction
Advancements in computer vision research have put the transformer architecture as the state-of-the-art in computer vision tasks. But one of the known drawbacks of the transformer architecture is the high number of parameters. This can lead to a more complex and inefficient algorithm. This paper aims to reduce the number of parameters and in turn, made the transformer more efficient. We present Sparse Transformer (SparTa) block, a modified transformer block with an addition of a sparse token converter that reduces the amount of tokens used. We use the SparTa block inside the Swin-T architecture (SparseSwin) to leverage Swin’s capability to downsample its input and reduce the number of initial token to be calculated. 

The proposed SparseSwin model outperforms other state-of-the-art models in image classification with an accuracy of 86.90%, 97.43%, and 85.35% on the ImageNet100, CIFAR10, and CIFAR100 datasets respectively, despite its fewer number of parameters. This result highlights the potential of a transformer architecture using a sparse token converter with a limited number of tokens to optimize the use of the transformer and improve its performance. 

## Architecture
<p align="center" ><img src="https://media.discordapp.net/attachments/449985531372240908/1117657023056728194/sparseswin.png?width=1440&height=288" width="768"/> </p>
<p align="center" ><img src="https://media.discordapp.net/attachments/449985531372240908/1117657023287410738/sparta_block.png?width=1163&height=662" width="768"/> </p>

# Results on ImageNet100, CIFAR10, and CIFAR100
<p align="center" ><img src="https://cdn.discordapp.com/attachments/449985531372240908/1117781192956182538/table.png" width="768"/> </p>



