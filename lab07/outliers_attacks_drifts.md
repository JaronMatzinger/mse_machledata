#  Detecting Outliers, Attacks, and Drift

In the previous part, we discussed serving a model. In this section, we will discuss how to protect our model in the real world. We will discuss how to detect outliers, adversarial attacks, and drift.

_Note: If you are in a hurry, you can obtain the weights for the models that we train in this part from [here](https://drive.switch.ch/index.php/s/7B5ksEnpmvIOQAn)._

## Outlier Detection

Outliers are data points that are different from other data points. They can be caused by errors in data collection or can be indicative of some new trends in the data. Outliers can significantly affect the performance of machine learning models. Therefore, it is important to detect and handle outliers.

While there are many ways to detect outliers for tabular data, we will discuss how to detect outliers in image data. For this, we will use Variational Autoencoders (VAEs). VAEs are generative models, so they learn the underlying distribution of the data. This is crucial for detecting outliers, since outliers are data points that are different from other data points - they are samples from the tails of the distribution or not even part of it.
If you have never heard of VAEs, you can read more about them [in this detailed introduction to variational autoencoders](https://arxiv.org/abs/1906.02691).

In practice, detecting outliers using VAEs amounts to checking how well the input data can be reconstructed by the model. If the input data cannot be reconstructed well, it is likely an outlier.

In [`notebooks/outlier_detection.ipynb`](notebooks/outlier_detection.ipynb), we will use the not-so-exciting MNIST dataset to detect outliers using a VAE to demonstrate the concept.

## Adversarial Attack Detection

In a previous lab, we discussed how to generate and defend against adversarial attacks. In this section, we will discuss how to detect adversarial attacks.
The method introduced below was proposed in [Adversarial Detection and Correction by Matching Prediction Distributions](https://arxiv.org/pdf/2002.09364.pdf).

If you think about it, adversarial attacks are not too different from outliers. So, you might be tempted to use the same approach to detect adversarial attacks as you would to detect outliers. However, there is an issue: (Variational) autoencoders are trained to find a transformation $T$ that reconstructs the input data $x$ as well as possible. This is done by minimizing the reconstruction error $L(x, T(x)) = \|x - x'\|^2$. However, these types of loss functions suffer from a fundamental flaw for the detection of adversarial attacks: they are not sensitive to small perturbations in the input data. This is because the loss function is minimized when the input data is reconstructed as well as possible, regardless of whether the input reconstruction error is due to an adversarial attack or not.

One way to detect adversarial attacks is to use a model-dependent reconstruction error. Given a model $M$, we can optimize the weights $\theta$ of the model to minimize the following loss function:

$$\min\limits_\theta D_{KL}(M(x) \| M(AE_\theta(x)))$$

$M$ is the model we want to protect from adversarial attacks - e.g. a classifier. During training of the autoencoder, the weights of $M$ are frozen, and we use its output probabilities to compute the loss. The loss function is the Kullback-Leibler divergence between the output probabilities of the model $M$ and the output probabilities of the model $M$ when the input data is reconstructed by the autoencoder. The intuition behind this loss function is that the output probabilities of the model $M$ should be similar when the input data is reconstructed by the autoencoder and when it is not. If the output probabilities are not similar, it is likely that the input data is an adversarial attack.

### Excursion: What is the Kullback-Leibler divergence?

We've been referring to the Kullback-Leibler divergence a lot, so let's take a moment to explain what it actually is and does. The Kullback-Leibler divergence is a measure of how one probability distribution $P$ differs from a second, reference probability distribution $Q$. Its definition depends on whether the distributions are discrete or continuous.

For discrete distributions $P$ and $Q$ over some sample space $\mathcal{X}$, the Kullback-Leibler divergence is defined as:

$$D_{KL}(P \| Q) = \sum\limits_{x \in \mathcal{X}} P(x) \log(\frac{P(x)}{Q(x)})$$

In other words, the Kullback-Leibler divergence is the expectation of the logarithmic difference between the probabilities of the two distributions, weighted by the probabilities of the first distribution. Note that there are equivalent definitions.

Similar definitions hold for continuous distributions. We will not go into the details here, but you can read more about the Kullback-Leibler divergence [here](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

### Back to the main topic

In [`notebooks/adversarial_attack_detection.ipynb`](notebooks/adversarial_attack_detection.ipynb), we will again use the MNIST dataset to detect adversarial attacks using a model-dependent reconstruction error.

## Outliers, attacks, and drift detection with MLServer

In practice, we have to integrate the detection of outliers, attacks, and drift into our serving pipeline. This can be done by monitoring the input data distribution and the model's performance over time. When you think back to the previous part about `MLServer`, you might see a few ways to integrate these detection mechanisms into the serving pipeline. For example, you could add a new endpoint to the server that returns the reconstruction error of the input data. If you use a custom inference runtime (as we did in the previous part), you could then call this endpoint in the `predict` method.

If you choose to use `alibi-detect`, a library that provides a wide range of outlier, adversarial attack, and drift detection algorithms, you can even rely on a [pre-built inference runtime](https://mlserver.readthedocs.io/en/latest/runtimes/alibi-detect.html)!
