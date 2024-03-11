# NC-TTT

Official repository of the CVPR 2024 paper "NC-TTT: A Noise Constrastive Approach for Test-Time Training", by David Osowiechi, Gustavo A. Vargas Hakim, Mehrdad Noori, Milad Cheraghalikhani, Ali Bahri, Moslem Yazdanpanah, Ismail Ben Ayed, and Christian Desrosiers.
The whole article can be found [here](https://openaccess.thecvf.com/content/ICCV2023/html/****html).
This work was greatly inspired by the code in [ClusT3]([(https://github.com/dosowiechi/ClusT3.git)]).

We propose a novel unsupervised TTT technique based on the discrimination of noisy feature maps. By learning to classify noisy views of projected feature maps, and then adapting the model accordingly on new domains, classification performance can be recovered by an important margin.

![Diagram](https://github.com/GustavoVargasHakim/NCTTT/blob/master/NC-TTT.png)

## Datasets

The experiments utilize the CIFAR-10 training split as the source dataset. It can be downloaded from 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), or can also be done using torchvision
datasets: `train_data = torchvision.datasets.CIFAR10(root='Your/Path/To/Data', train=True, download=True)`.
The same line of code can be used to load the data if it is already downloaded, just by changing the
argument `download` to `False`.

At test-time, we use CIFAR-10-C and CIFAR-10-new. The first one can be downloaded from [CIFAR-10-C](
https://zenodo.org/record/2535967#.YzHFMXbMJPY). For the second one, please download the files 
`cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy` from [CIFAR-10-new](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets).
All the data should be placed in a common folder from which they can be loaded, e.g., `/datasets/`.

The training works the same way on CIFAR-100 dataset and it can be downloaded from [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).
At test-time, we use CIFAR-100-C which can be downloaded from [CIFAR-100-C](https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1).

## Citation

If you found this repository, or its related paper useful for your research, you can cite this work as:

```
@inproceedings{NCTTT2024,
  title={NC-TTT: A Noise Constrastive Approach for Test-Time Training},
  author={David Osowiechi and Gustavo A. Vargas Hakim and Mehrdad Noori  and Milad Cheraghalikhani and Ali Bahri  and Moslem Yazdanpanah and Ismail Ben Ayed and Christian Desrosiers},
  booktitle={***},
  pages={},
  month={June},
  year={2024}
}
```

