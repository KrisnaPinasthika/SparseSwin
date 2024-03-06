# SparseSwin

---

This repo is the official implementation of <a href="https://doi.org/10.1016/j.neucom.2024.127433">SparseSwin: Swin Transformer with Sparse Transformer Block </a>. <br>

<The>SparseSwin is an architecture designed for addressing image classification cases, but it is not limited to object detection and image segmentation scenarios. SparseSwin is constructed by employing the Swin Transformer as the primary architecture, implementing our proposed Sparse Transformer (SparTa) Block in the fourth stage.The SparTa Block possesses computational complexity unrelated to the size of the input image, allowing it to efficiently handle large input image sizes.</p>

<p>In detail, the architecture used in this research is illustrated in Figures 1 and 2.</p>
<figure>
    <center>
        <img src="./Sources/fig1 sparseswin.png">
        <figcaption><b>Fig. 1</b> The architecture of SparseSwin</figcaption>
    </center>
</figure>

<figure>
    <center>
        <img src="./Sources/fig2 sparta block.png">
        <figcaption><b>Fig. 2</b> The successive SparTa blocks in stage 4 of SparseSwin for image classification</figcaption>
    </center>
</figure>

<figure>
    <center>
        <img src="./Sources/fig3 complexity calculation.png">
        <figcaption><b>Fig. 3</b> Graph Comparison of computational complexity between Swin Transformer Block and our proposed SparTa Block assuming the input image is a square so that the height and width of the features have the same value. In this figure, the x-axis represents the height and width of the input features in block 4 of the Swin Transformer and SparseSwin architectures while the y-axis represents the computational complexity of the attention calculation.</figcaption>
    </center>
</figure>

<p>Through this research, we obtained accuracy improvements on several transformer architectures with similar parameter sizes on benchmark datasets such as ImageNet100, CIFAR10, and CIFAR100. </p>

<figure>
    <img src="./Sources/table4.png">
    <img src="./Sources/table5.png">
    <img src="./Sources/table6.png">
</figure>
