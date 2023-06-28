# Super-Resolution with Iterative Kernel Correction (IKC)

## Introduction
In today's society, where vast amounts of image information are handled, Super-Resolution (SR), which enhances the resolution of images, is an important technology. Among the techniques in the field of SR, methods based on deep learning are gaining popularity due to their efficiency and effectiveness in problem-solving. Most of these deep learning-based methods assume or predefine the blur kernel during the down-sampling process. However, in many cases, we do not have knowledge of the blur kernel for the images we handle. Therefore, SR methods with additional processes to estimate the blur kernel, such as Iterative Kernel Correction (IKC) and Deep Alternating Network (DAN), have been proposed. IKC utilizes the kernel mismatch phenomenon observed in SR images to estimate the blur kernel by iteratively applying a process that calculates the difference between the actual kernel and the predicted kernel. 

## Objective
This research aims to address the drawback of IKC, which involves lengthy inference time due to the repetitive execution of the model. To overcome this issue, we propose a new model that merges the modules of IKC and reduces the number of iterations.

## Proposed Model
The proposed model combines the modules of IKC while optimizing the iteration process to minimize inference time. By integrating the iterative kernel correction and reducing the number of iterations required, we aim to achieve efficient and effective super-resolution with accurate blur kernel estimation.

## Results
We anticipate that the proposed model will significantly reduce inference time while maintaining or improving the accuracy of blur kernel estimation and super-resolution performance. Experimental results and comparisons will be presented in the final research report.

## Conclusion
The new model, which merges the modules of IKC and reduces the number of iterations, addresses the issue of lengthy inference time while maintaining the effectiveness of super-resolution with accurate blur kernel estimation. This research aims to contribute to the advancement of super-resolution techniques and their practical application in various domains where image enhancement is crucial. Further improvements and optimizations can be explored in future studies.
