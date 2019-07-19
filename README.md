# BERT-Activation-Function

Transformer에서는 activation function을 ReLU를 사용하였지만 BERT에서는 보다 부드러운 형태의 GELU activation function을 사용하였습니다. 이는 음수에 대해서는 미분이 가능해 약간의 Gradient를 전달할 수 있습니다. 아래는 Gelu


<img src = "https://user-images.githubusercontent.com/1250095/50040221-c9a08700-0082-11e9-8aec-8b11d35ab616.png" width=50%>


## GELU 


```python

def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

```

## Swish
```python

def swish(x):
  return x * tf.nn.sigmoid(x)

```

## Swish beta
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
[Swish in depth: A comparison of Swish & ReLU on CIFAR-10](https://medium.com/@jaiyamsharma/swish-in-depth-a-comparison-of-swish-relu-on-cifar-10-1c798e70ee08)
