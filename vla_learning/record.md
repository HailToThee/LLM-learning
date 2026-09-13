## AGC
### attack_type: c&w
**flowers102**
📊 Local Linear TTA [emp] |  Attack: PGD ε=4/255  |  2463 samples
  Clean Accuracy (Raw):       63.99%  (1576/2463)
  Adv Accuracy (Raw):         17.01%  (419/2463)
  Clean Accuracy (Defense):   62.04%  (1528/2463)
  Adv Accuracy (Defense):     58.95%  (1452/2463)

**caltech101**
📊 Local Linear TTA [emp] |  Attack: PGD ε=4/255  |  2465 samples
  Clean Accuracy (Raw):       91.36%  (2252/2465)
  Adv Accuracy (Raw):         41.10%  (1013/2465)
  Clean Accuracy (Defense):   91.85%  (2264/2465)
  Adv Accuracy (Defense):     89.49%  (2206/2465)


### deepfool
**flowers102**
  Clean Accuracy (Raw):       63.99%  (1576/2463)
  Adv Accuracy (Raw):         25.42%  (626/2463)
  Clean Accuracy (Defense):   62.12%  (1530/2463)
  Adv Accuracy (Defense):     58.71% 5 (1446/2463)
**caltefch101**
  Clean Accuracy (Raw):       91.36%  (2252/2465)
  Adv Accuracy (Raw):         41.10%  (1013/2465)
  Clean Accuracy (Defense):   91.85%  (2264/2465)
  Adv Accuracy (Defense):     89.66%  (2210/2465)

### mi-fgsm
  Clean Accuracy (Raw):       91.36%  (2252/2465)
  Adv Accuracy (Raw):          0.00%  (0/2465)
  Clean Accuracy (Defense):   92.13%  (2271/2465)
  Adv Accuracy (Defense):     99.31%  (2448/2465)


  Clean Accuracy (Raw):       63.99%  (1576/2463)
  Adv Accuracy (Raw):          0.00%  (0/2463)
  Clean Accuracy (Defense):   62.97%  (1551/2463)
  Adv Accuracy (Defense):     87.21%  (2148/2463)





TTP
caltech101:
mi-fgsm:83.99%
cw:58.02%
deepfool:60.07%


mi_fgsm44.24%


python padding_refine_1.py --dataset caltech101 --device cuda:4 --attack cw

r-tpt

c&w

caltech101
Original Accuracy: 89.37%
CW Attack Accuracy:
  No Padding: 44.22%
  Trained Padding: 54.12%