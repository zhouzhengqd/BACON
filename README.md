<p align="center">
<img src="Fig/logo.png" width="100%" class="center" alt="pipeline"/>
</p>
### [Paper](https://arxiv.org/pdf/2406.01112) | [Distilled Datasets](https://share.multcloud.link/share/f496af96-494a-4815-a7c9-e93cd95ecdd1)
This repository contains the code and implementation for the paper "**BACON: Bayesian Optimal Condensation Framework for Dataset Distillation**".
## 👨‍💻 Authors

- [Zheng Zhou](https://zhouzhengqd.github.io/)<sup>1</sup>, [Hongbo Zhao](https://shi.buaa.edu.cn/09698/zh_CN/index.htm)<sup>1</sup>, [Guangliang Cheng](https://sites.google.com/view/guangliangcheng)<sup>2</sup>, [Xiangtai Li](https://lxtgh.github.io/)<sup>3</sup>, [Shuchang Lyu*](https://scholar.google.com/citations?user=SwGcxzMAAAAJ&hl=en)<sup>1</sup>, [Wenquan Feng](https://shi.buaa.edu.cn/fengwenquan/zh_CN/index/132879/list/)<sup>1</sup>, and [Qi Zhao](https://shi.buaa.edu.cn/07297/zh_CN/index.htm)<sup>1</sup> (* Corresponding Author)
- <sup>1</sup>[Beihang Univerisity](https://www.buaa.edu.cn/), <sup>2</sup>[University of Liverpool](https://www.liverpool.ac.uk/), and <sup>3</sup>[Nanyang Technological University](https://www.ntu.edu.sg/)
  
📧 For inquiries, please reach out via email: zhengzhou@buaa.edu.cn. Feel free to ask any questions!
## 🔍 Overview
<p align="center">
<img src="./Fig/overview.png" width=100% height=55.2%
class="center">
  <figcaption><strong>Figure 1:</strong> Comparison of BACON and existing DD methods: (a) Traditional methods align gradients and distributions on original and synthetic datasets. (b) BACON transforms DD into a Bayesian optimization task, generating synthetic images using likelihood and prior probabilities.</figcaption>
</p>
 
> **Abstract** Dataset Distillation (DD) reduces dataset size while maintaining test set performance, helping to cut storage and training costs. Current DD methods struggle with large datasets and lack a solid theoretical foundation. To address this, we introduce the <u>**BA**</u>yesian Optimal <u>**CON**</u>densation Framework (<u>**BACON**</u>), the first Bayesian approach to DD. BACON formulates DD as a minimization problem in Bayesian joint distributions and derives a numerically feasible lower bound. Our experiments show that BACON outperforms state-of-the-art methods, with significant accuracy improvements on CIFAR-10 and TinyImageNet. BACON seamlessly integrates with existing systems and boosts DD performance. Code and distilled datasets are available at [BACON](https://github.com/zhouzhengqd/BACON).

## 🚀 Contributions
<p align="center">
<img src="./Fig/method.png" width=100% height=55.2%
class="center">
  <figcaption><strong>Figure 2:</strong> Illustration of BACON: The neural network outputs a distribution from both synthetic and real datasets. BACON formulates this distribution as a Bayesian optimal condensation risk function and derives its optimal solution using Bayesian principles.</figcaption>
</p>

- **First Bayesian DD Framework:** Introduces the Bayesian framework to dataset distillation, providing a theoretical basis for improved performance.

- **Efficient Distillation:** Formulates DD as minimizing an expected risk function in joint distributions.

- **Theoretical Lower Bound:** Derives a feasible lower bound for the distillation process.

- **Superior Performance:** Outperforms existing methods on multiple benchmarks and can be easily integrated into existing systems.

## 📈 Experimental Results
The distilled datasets are available at [Distilled Dataset](https://share.multcloud.link/share/f496af96-494a-4815-a7c9-e93cd95ecdd1).
<!-- (https://drive.google.com/drive/folders/1hZCowM21nfSOkRtm8VuK1lEpP7Bd1jCq?usp=sharing). -->
### Comparison to the State-of-the-art Methods
- **IPC-50**

| Method | MNIST | Fashion-MNIST | SVHN | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
| :------: | :-----:  | :----: | :-----: | :----: |:----: |:----: | 
| **DM** | 94.8 | - | - | 63 | 43.6 | - |
| **IDM** | 97.01 | 84.03 | 87.5 | 67.5 | 50 | - |
| **BACON** | 98.01 | 85.52 | 89.1 | 70.06 | 52.29 | - | 
- **IPC-10**

| Method | MNIST | Fashion-MNIST | SVHN | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
| :------: | :-----:  | :----: | :-----: | :----: |:----: |:----: | 
| **DM** | 97.3 | - | - | 48.9 | 29.7 | 12.9 |
| **IDM** | 96.26 | 82.53 | 82.95 | 58.6 | 45.1 | 21.9 |
| **BACON** | 97.3 | 84.23 | 84.64 | 62.06 | 46.15 | 25 | 
- **IPC-1**

| Method | MNIST | Fashion-MNIST | SVHN | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
| :------: | :-----:  | :----: | :-----: | :----: |:----: |:----: | 
| **DM** | 89.2 | - | - | 26 | 11.4 | 3.9 |
| **IDM** | 93.82 | 78.23 | 69.45 | 45.60 | 20.1 | 10.1 |
| **BACON** | 94.15 | 78.48 | 69.44 | 45.62 | 23.68 | 10.2 | 
### Visulizations
<!-- ![image samples](./Fig/visulization.png) -->
![image samples](./Fig/mnist.png)
![image samples](./Fig/f-mnist.png)
![image samples](./Fig/svhn.png)
<!-- ![image samples](./Fig/cifar-100.png) -->
## 🚀 Getting Started
### Step 1
- Run the following command to download the Repo.
  ```
  git clone https://github.com/zhouzhengqd/BACON.git
  ```
### Step 2
- Download Datasets (MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, Tiny-ImageNet). 
<!-- - at [Dataset](https://drive.google.com/drive/folders/1hZCowM21nfSOkRtm8VuK1lEpP7Bd1jCq?usp=sharing). -->
### Step 3
- Run the following command to create a conda environment
    ```
    cd BACON
    cd Code
    conda env create -f environment.yml
    conda activate bacon
    ```
## 📁 Directory Structure
- BACON
    - Code
        - data
          - datasets
        - checkpoints
        - result
        - Files for BACON
        - enviroment.yml
        - ...
        - ...
        - ...

## 🛠️ Command for Reproducing Experiment Results and Evaluation
- For example: Validate on the MNIST, other datasets follow the "Command.txt" file.
- BACON MNIST IPC-50
  ```
    python3 -u BACON_mnist.py --dataset MNIST --model ConvNet --ipc 50 --dsa_strategy color_crop_cutout_flip_scale_rotate --init real --lr_img 0.2 --num_exp 5 --num_eval 5 --net_train_real --eval_interval 100 --outer_loop 1 --mismatch_lambda 0 --net_decay --embed_last 1000 --syn_ce --ce_weight 0.5 --train_net_num 1 --aug
  ```
- BACON MNIST IPC-10
  ```
    python3 -u BACON_mnist.py --dataset MNIST --model ConvNet --ipc 10 --dsa_strategy color_crop_cutout_flip_scale_rotate --init real --lr_img 0.2 --num_exp 5 --num_eval 5 --net_train_real --eval_interval 100 --outer_loop 1 --mismatch_lambda 0 --net_decay --embed_last 1000 --syn_ce --ce_weight 0.5 --train_net_num 1 --aug
  ```
- BACON MNIST IPC-1
  ```
    python3 -u BACON_mnist.py --dataset MNIST --model ConvNet --ipc 1 --dsa_strategy color_crop_cutout_flip_scale_rotate --init real --lr_img 0.2 --num_exp 5 --num_eval 5 --net_train_real --eval_interval 100 --outer_loop 1 --mismatch_lambda 0 --net_decay --embed_last 1000 --syn_ce --ce_weight 0.5 --train_net_num 1 --batch_real 5000 --net_generate_interval 5 --aug
  ```
## 🙏 Acknowledge
We gratefully acknowledge the contributors of DC-bench and IDM, as our code builds upon their work ([DC-bench](https://github.com/justincui03/dc_benchmark?tab=readme-ov-file) and [IDM](https://github.com/uitrbn/IDM)).
## 📚 Citation
```
@article{zhou2024bacon,
  title={BACON: Bayesian Optimal Condensation Framework for Dataset Distillation},
  author={Zhou, Zheng and Zhao, Hongbo and Cheng, Guangliang and Li, Xiangtai and Lyu, Shuchang and Feng, Wenquan and Zhao, Qi},
  journal={arXiv preprint arXiv:2406.01112},
  year={2024}
}
```
## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zhouzhengqd/BACON&type=Date)](https://star-history.com/#zhouzhengqd/BACON&Date)