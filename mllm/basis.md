# Basis of Multimodal Learning
**global**:
let $f_\theta$ represents the image encoder, and $g_\phi$ represents the text encoder

**Multimodal tasks**:
1. VQA: Visual Question Answering. Given an image, ask question about the image's content.
2. VR: Visual Reasoning(what is going to be inferenced according to the image)/ Retrival(which class is the object?)
3. VE: Visual Entailment(if the image support this comment?)

### Difference bewteen Cross-entropy and KL Divergence:
#### CrossEntropy:
$$H(P,Q) = -\sum_{x}P(x)\log Q(x)$$
how worse it could be by using model distribution $Q$ to approximate the true distribution $P$, and minimize the cross-entropy loss to make $Q$ closer to $P$.
```
used for:
1. classification tasks
2. language modeling regarding the next token prediction
3. ITC/softmax contrasive learning loss
```
#### KL Divergence:
$$D_{KL}(P||Q) = \sum_{x}P(x)\log \frac{P(x)}{Q(x)}$$
measure the difference between two distributions $P and Q$, and minimize the KL divergence to make $Q$ closer to $P$.
```
used for:
1. knowledge distillation
2. variational autoencoders
```
$$H(P,Q) = H(P)+D_{KL}(P||Q)$$

###  Loss
#### ITC: image to text loss
assume a batch contains N pairs image-to-text samples
$$\mathcal{B} = \{I_i, T_i\}^N_{i = 1}$$
where $I_i$ represents image_i, $T_i$ represents text_i.
ITC loss: make corresponding image-text closer, otherwise farther.
 $$s_{ij} = f_\theta(I_i)^\top g_\phi(T_j)$$
 $$\mathcal{L}_{ITC} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^N \exp(s_{ij}/\tau)}$$
where $\tau$ is a temperature hyperparameter.
<!-- **actually**, $s_{ij}$ is just one type of mixing multimodal features, there are more ways to mix multimodal features, even **Transformer encoder** arcitecture. -->

#### ITM: image to text matching loss
ITM loss: given an image and a text, determine whether they match or not. (Optimizing with a easy classification is not good, use the hardest negative samples to train is better.*ALBEF*)

is more like a binary classification problem, we can use cross-entropy loss to train the model.

**actually** ITM is used in **cross-modal encoder(what i have mentioned above)** to train the model.

let image token:
$$X_i = \{x_{i,1}, x_{i,2}, \ldots, x_{i,M}\}$$
let text token:
$$Y_j = \{y_{j,1}, y_{j,2}, \ldots, y_{j,N}\}$$
then we can concatenate the image token and text token together:
$$h_{ij} = F_{\Psi}(X_i, Y_j)$$
usually take [CLS] token as the representation of the whole sequence, then we can use a linear layer to predict whether they match or not:
$$p_{ij} = \sigma(W h_{ij}^{[CLS]} + b)$$
$$\mathcal{L}_{ITM} = -\frac{1}{\mathcal{P}}\sum_{i,j} y_{ij} \log p_{ij} + (1-y_{ij}) \log (1-p_{ij})$$

#### MLM: masked language modeling loss
MLM loss: given a text, mask some tokens and predict the masked tokens, which is like Bert's pretraining task.
let the original text token:
$$Y = \{y_1, y_2, \ldots, y_N\}$$
after masking some tokens, we get:
$$\hat{Y} = \{\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_N\}$$
then we can use the cross-entropy loss to train the model:
$$\mathcal{L}_{MLM} = -\frac{1}{N}\sum_{i=1}^N \log p(\hat{y}_i | \hat{Y}_{\backslash i}, X)$$
where $\hat{Y}_{\backslash i}$ represents the masked text token except the $i$-th token, and $X$ represents the image token. 

#### LM: language modeling loss
LM loss: given a text, predict the next token, which is like GPT's pretraining task.
let the original text token:
$$Y = \{y_1, y_2, \ldots, y_N\}$$
then we can use the cross-entropy loss to train the model:
$$\mathcal{L}_{LM} = -\frac{1}{N}\sum_{i=1}^N \log p(y_i | Y_{<i}, X)$$
where $Y_{<i}$ represents the text token before the $i$-th token, and $X$ represents the image token.