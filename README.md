# A3T
Code for ECML-PKDD 2023 "A3T: Accuracy Aware Adversarial Training"

## Usage
### CIFAR-10

```
python train_resnet.py # to test with ResNet18
```
```
python train_wideresnet.py # to test with WideResNet
```

### GLUE 

```
python glue.py $dataset $adv_method
```
- The dataset parameter can take a value of one of the GLUE datasets. 
- The adv_method parameter must be one of the "clean", "AT", and "A3T" as an adversarial attack method.
Each dataset should first run with a 'clean' argument, then one of 'AT' or 'A3T' can be used.

## Citing this work
If you use this code in your work, please cite the accompanying paper:

```
@article{altinisik2022a3t,
  title={A3T: Accuracy Aware Adversarial Training},
  author={Altinisik, Enes and Messaoud, Safa and Sencar, Husrev Taha and Chawla, Sanjay},
  journal={arXiv preprint arXiv:2211.16316},
  year={2022}
}
```
The vision part of the code is forked from the repository of YisenWang/MART.