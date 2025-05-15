Project with custom CUDA-based optimizations of different NNs, implemented in pytorch.

Currently supported NNs and ops:

1) Resnet
- relu op
- batchnorm 2d op
- fused bn + relu

fusion performance results:

```
Using custom ops:  []
Top predictions:
King Charles Spaniel: 98.47%
       Japanese Chin: 0.37%
Welsh Springer Spaniel: 0.34%
    Brittany Spaniel: 0.20%
            Papillon: 0.17%
Inference avg time: 264.36 ms
```

```
Using custom ops:  ['fused_bn_relu']
replacing torch ops with custom kernels!
fused bn and relu
Top predictions:
King Charles Spaniel: 98.47%
       Japanese Chin: 0.37%
Welsh Springer Spaniel: 0.34%
    Brittany Spaniel: 0.20%
            Papillon: 0.17%
Inference avg time: 261.09 ms
```

