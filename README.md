# BERT-Activation-Function

Transformer에서는 activation function을 ReLU를 사용하였지만, BERT에서는 보다 부드러운 형태의 GELU activation function을 사용하였습니다. 이는 ReLU에 비해 음수에 대해서도 미분 가능하여, 약간의 Gradient를 전달할 수 있습니다. 아래는 BERT에서 사용한 GELU 구현 코드입니다.

```python

def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

```

[Searching for Activation Functions](https://arxiv.org/abs/1710.05941) 논문에서는 Machine Translation Task에서의 Activation function 성능 비교를 보여주고 있습니다. 해당 논문을 참고하여, 본 repository에서는 SWISH activation function과 GELU의 성능 비교를 진행하였습니다. 

<img src = "https://k.kakaocdn.net/dn/IvZvO/btquhj4JtWW/x42RsvOWqfxvqkkcAijd1k/img.png" width=65%>

## SWISH activation function
```python

def swish(x):
  return x * tf.nn.sigmoid(x)

```

## SWISH BETA activation function
```python

def alt_swish(x):
  beta = tf.Variable(initial_value=1.0, trainable=True, name='swish-beta')
  return x * tf.nn.sigmoid(beta*x)
```




## Step별 성능 비교
Base Model 기준으로 총 100만 step을 학습을 진행하였고 기존 GELU로 구축한 Model과 성능 비교 결과는 아래와 같습니다. 측정 기준은 step별 KorQuAD Task의 F1,EM으로 측정하였습니다.
<br>

* Base Model(12-layer, 768-hidden, 12-heads)<br>

| Step | seq_length | GELU F1 | GELU EM | swish beta F1 | swish beta EM | swish F1 | swish EM |
|:-------:|:-------:|:-------:| :-------:| :-------:| :-------:| :-------:| :-------:| 
| 40만 | 128 | 77.41% | 62.12% | 00.00% | 00.00% |
| 60만 | 128 |  78.63% | 63.17% | 00.00% | 00.00% |
| 90만 | 128 |  80.98% | 65.25% | 00.00% | 00.00% |
| 100만 | 512 | 91.40% | 79.47% | 00.00% | 00.00% |




## Reference

[Swish, a new activation function for Neural Network](https://jmlb.github.io/ml/2017/12/31/swish_activation_function/)<br>
[Swish in depth: A comparison of Swish & ReLU on CIFAR-10](https://medium.com/@jaiyamsharma/swish-in-depth-a-comparison-of-swish-relu-on-cifar-10-1c798e70ee08)<br>
[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)<br>
[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
