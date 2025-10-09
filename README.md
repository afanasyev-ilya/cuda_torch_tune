Project with custom CUDA-based optimizations of different NNs, implemented in pytorch.

Currently supported NNs and ops:

### 1. Resnet
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


### 2. Attention

```
Q/K/V shape: torch.Size([4, 4096, 128])
Inference (torch mha) min time: 6.65 ms
Inference (torch mha) avg time: 6.80 ms
Inference (torch mha) max time: 7.26 ms
```

```
Inference (layerwise attention) min time: 5.88 ms
Inference (layerwise attention) avg time: 5.92 ms
Inference (layerwise attention) max time: 5.93 ms
```

```
Inference (cuda naive) min time: 21.96 ms
Inference (cuda naive) avg time: 22.29 ms
Inference (cuda naive) max time: 22.96 ms
```

1. cublas GEMMs fused with transpose + warp-level reduction softmax
```
Inference (cuda opt layerwise) min time: 3.92 ms
Inference (cuda opt layerwise) avg time: 4.34 ms
Inference (cuda opt layerwise) max time: 4.94 ms
```

2. List other optimizations
TODO