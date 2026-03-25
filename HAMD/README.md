# HAMD-RSISR: Hybrid Attention and Multi-Dictionary for Remote Sensing Super-Resolution

This job opportunity arises from the paper "HAMD-RSISR: Hybrid Attention and Multi-Dictionary for Remote Sensing Super-Resolution",which has been published on IEEE JSTARS.The improvement is primarily based on "Transcending the Limit of Local Window Advanced Super-Resolution Transformer" from CVPR2024. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Transcending_the_Limit_of_Local_Window_Advanced_Super-Resolution_Transformer_with_CVPR_2024_paper.html)] [[arXiv](https://arxiv.org/abs/2401.08209)]
Given my limited coding proficiency, I am submitting this as a record. It should be noted that this code cannot be used for professional learning or work



> **Abstract:** Remote sensing image super-resolution aims to enhance low-resolution remote sensing images to restore highquality details. Currently, methods based on Swin Transformer have achieved remarkable success due to their excellent contextual attention capabilities and moving window mechanism. However, the limitations of this window mechanism make it difficult for the model to achieve cross-window information interaction, resulting in the loss of important complementary information. To address this issue, inspired by the Adaptive Token Dictionary (ATD), we propose a new remote sensing image super-resolution method, HAMD-RSISR, and for the first time, introduce the use of a “multi-dictionary mechanism” to break through the limitations of traditional window mechanisms. Unlike the single-dictionary ATD, our multi-dictionary mechanism enhances the network’s global context modeling capabilities at both coarse and fine granularities. Specifically, one dictionary is used for preliminary re-division of feature maps, while the other dictionary performs further refinement on the previous dictionary to process feature maps at deeper scales. In addition, we propose a hybrid attention mechanism to further effectively utilize channel, spatial, and frequency information in remote sensing images. Finally, the hybrid attention and multi-dictionary mechanisms process the input feature maps in parallel and perform feature fusion. Experimental results demonstrate the excellent performance of our HAMD in remote sensing image super-resolution. On the CLRS×2 dataset, it achieves a PSNR improvement of approximately 0.44dB compared to the state-ofthe-art RGT in recent years, and on the Uc Merced×4 dataset, it achieves a PSNR improvement of approximately 0.4dB compared to HAT.


## Environment
- Python 3.9
- PyTorch 2.0.1



## Citation

```
 """@article{xie2025hamd,
  title={HAMD-RSISR: Hybrid Attention and Multi-Dictionary for Remote Sensing Super-Resolution},
  author={Xie, Z and Wang, J and Song, W and others},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  publisher={IEEE}
}""

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and Transcending the limit of local window:
Advanced super-resolution transformer with adaptive token dictionary(ATD).

