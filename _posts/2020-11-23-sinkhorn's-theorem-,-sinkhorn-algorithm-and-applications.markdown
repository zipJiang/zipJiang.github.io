---
layout: posts
title:  "Sinkhorn's Theorem, Sinkhorn Alorithm and Applications"
---

# Sinkhorn's Theorem, Sinkhorn Algorithm and Applications

In this post I'm going to briefly discuss a computationally efficient way to calculate an approximate matrix factorization, called Sinkhorn's algorithm, and it's example application in machine learning.

## Sinkhorn's Theorem and Sinkhorn-Knopp Algorithm

The Sinkhorn's theorem states that every square matrix with positive elements can be transformed into a doubly stochastic matrix $$D_1AD_2$$. Where $$D_1$$ and $$D_2$$ are diagonal matrices with all positive main diagonals. The matrices $$D_1$$ and $$D_2$$ are themselves unique up to a constant factor.

Sounds simple, except the problem that what is a doubly stochastic matrix? Well, a doubly stochastic matrix is a all non-negative square matrix that is both row-normalized and column-normalized. To speak about this mathematically for a non-negative square matrix $$A \in \mathcal{R}^{N \times N}$$, $$\sum_{i = 1}^{N} a_{ij} = 1, \forall j \in [1, N]$$. Also, $$\sum_{j = 1}^{N} a_{ij} = 1, \forall i \in [1, N]$$.

Notice that to apply the Sinkhorn's theorem, we need to make sure that first, the matrix is a square matrix, and second, the matrix should be contain no negative element. As we should see later in this post, this property is satisfied by many of the problem that can be actually framed into calculating a corresponding doubly stochastic matrix. A set of non-negative integers summing into one has many direct interpretations that makes Sinkhorn's algorithm very powerful, like for
exmaple, probabilities, or, as we will see, a soft sorting indices.

What is great about Sinkhorn's theorem is that there is a efficient algorithm to calculate an approximation of this doubly stochastic matrix that has linear convergence. And the algorithm is very simple: one just iteratively normalize the matrix along rows and columns.

```python3
def sinkhorn(A, N, L):
    # Pseudo-Code for calculating the doubly stochastic matrix
    # using Sinkhorn-Knopp algorithm.
    # ----------
    # Input: positive matrix A[N x N], max iteration L

    for i in range(L):
        A = A / np.matmul(A, np.ones(N, 1))
        A = A / np.matmul(np.ones(1, N), A)

        # Test for convergence and early stop.
        if _converge:
            break

    return A
```

It's obvious that the algorithm has a complexity of $$O(N^2)$$ if applied a constant times at every step. Also notice that as we could write normalization as matrix multiplication, Sinkhorn algorithm is fully differentiable.

## Optimal Transport

We are now considering a very interesting application of this algorithm, which is in calculating optimal transport distance. Optimal transport is useful as a distance metric between distributions, in which way it plays a very important role in GAN and \*AE to stablize training.

So how is calculating a doubly stochastic matrix related to optimal transport?

First notice that optimal transport problem could be expressed in matrix form. Suppose we have $$N$$ number of starting points and $$N$$ number of destinations. Also, we have a distance matrix $$M \in \mathcal{R}^{N \times N}$$ where entry $$m_{ij}$$ indicates the distance, or in another word, the cost of moving items from starting point $$i$$ to destination $$j$$.

Now suppose our cargos are randomly distributed according to $$r$$ among the $$N$$ starting points. And we would like them to be distributed according to another distribution $$c$$. Now we can write up a matrix $$A \in \mathcal{R}^{N \times N}$$, where $$a_{ij}$$ corresponds to how much portion of the total cargos we would like to transport from starting point $$i$$ to destination $$j$$. Another way to view this matrix $$A$$ is that $$A$$ defines a joint distribution in the
transportation space and $$r$$ and $$c$$ could be seen as its marginals of starting points and destinations respectively. Now it's not hard to see that the final transportation cost would be the element-wise product between $$A$$ and $$M$$, or in other words the Frobenius inner product $$\langle A,M \rangle$$. In fact, this is called the Kantorovich relaxation of OT.

There is the conclusion that if $$M$$ is a distance matrix, that is, every element in the matrix comform to the three property of distance, the inner product is a distance itself. Confirming this should not be hard. To make a transfort optimal is to find the minimum cost of the total transport, that is, to minimize the Frobenius inner product.

I would like to stop and mention that as we now interpret $$A$$ as a joint probability matrix, we can define its entropy, the marginal probabiilty entropy, and KL-divergence between two different transportation matrix. These takes the form of

$$
    H(A) = \sum_{ij} a_{ij}\log_{ij}\\
    H(r) = \sum_{i = 1}^{N} (\sum_{j = 1}^{N}a_{ij})\log \sum_{j = 1}^{N}a_{ij}\\
    H(c) = \sum_{j = 1}^{N} (\sum_{i = 1}^{N}a_{ij})\log \sum_{i = 1}^{N}a_{ij}\\
    \mathcal{D}_{\text{KL}}(A\|B) = \sum_{ij} a_{ij}\log \frac{a_{ij}}{b_{ij}}
$$

Notice that as these are plain probability distributions, the inequality for joint probabilities still hold here:

$$
    H(A) \leq H(r) + H(c)
$$

This inequality is tight, as when $$r$$ is independent from $$c$$, the equality holds.

You might think that we are applying the Sinkhorn algorithm to solve the original OT problem as mentioned above, but actually no. original OT is notoriously sparse and since the optimization is non-convex it is not guaranteed to converge to a unique solution. Instead, researchers try to deal with this kind of sparsity with regularization terms that encourage diversity of different transportation route. Thus the objective of OT becomes:

$$
    P = \arg\min_{A \in U(r, c)} \langle A, M \rangle - \frac{1}{\lambda} H(A)
$$

It is known in the field of OT that the solution to this kind of $$\arg\min$$ problem is a rescaling of the matrix $$K = -\lambda M$$, which could be expressed as $$\tilde{K} = \text{Diag}(u)K\text{Diag}(v)$$, and with the following constraints:

$$
    u \circ (\tilde{K}\cdot v) = r\\
    (u\cdot\tilde{K}) \circ v = c
$$

This kind of rescaling problem could directly solved by Sinkhorn iteration. Notice that we are not using the exact form of Sinkhorn algorithm above, as now we have set of constraints regularizing the marginals $$r, c$$, but it can still be solved by mapping the row-marginal to $$r$$ and column-marginal to $$c$$ respectively.

## From Wasserstein Autoencoder to Sinkhorn Autoencoder

To get a sense of why this kind of OT is useful in machine learning, we now look at a example in the autoencoder literature. Learning meaningful low dimension representation of surface data often needs proper regularization to the hidden states. We know that in VAE, there is a KL-divergence term pulling $$ Q_{\phi}(Z\mid x) $$ to $$ P_{\theta}(Z) $$, where as in AAE, people use a discriminator to align the marginal distribution $$ \int Q_{\phi}(Z \mid X)P_{\text{data}}(X)dX $$ with $$ P_{\theta}(Z) $$.

Wasserstein autoencoder is a theoretical formulation of modeling the hidden space by aligning the modeling distribution $$P_{\theta}(X)$$ target data distribution $$P_{\text{data}}(X)$$ when given a specific generation function $$G: Z \rightarrow X$$ from a prior hidden space distribution $$p(Z)$$. Basically, the authors show that, minimizing the Wasserstein distance $$W_c$$ in the data space when giving a deterministic decoder $$G$$ as mentioned before, is equivalent to minimizing the expected difference between real datapoints $$X$$ and their encoded and decoded counter-parts $$G(Z)$$, with the constraints that the marginal distribution of $$ \int Q(Z \mid X)P(X)dX $$ is the same as the prior $$P(Z)$$. A relexation of this would give us the objective:

$$
    \mathcal{D}_{WAE} = \inf_{Q(Z \mid X) \in Q} \mathbb{E}_{P_{\text{data}}(X)}\mathbb{E}_{Q(Z \mid X)}[c(X, G(Z))] + \lambda \mathcal{D}_{Z}(Q(Z), P(Z))
$$

Because Wasserstein GAN approaches this formulation through empirical relaxation, they lack a rigorous interpretation why adding a divergence term on $$Q(Z), P(Z)$$ should work and what kind of divergence should work best. In fact, if we choose different divergence we are arriving at different kind of autoencoder formulation. For example, using adverserial loss for this term will give us adversarial autoencoder.

Sinkhorn autoencoder reaffirm this kind of autoencoder formulation but with a stronger theoretical intuition. They show that by decomposing the Wasserstein-p loss at the LHS when restricting $$ Q(Z \mid X)P(X) = P(Z) $$, we will have the exact objective as mentioned by WAE, but to directly optimize this objective, we should use a specific kind of divergence for $$Q(Z)$$ and $$P(Z)$$, namely the same Wasserstein-p distance. Now we are presented with exactly the same problem as we have seen in the OT section, with a prescribed cost matrix $M$ of $$l_p$$ distance. And directly optimizing this objective is difficult so.... That's exactly where Sinkhorn algorithm again comes to the rescue. First notice that relaxed Sinkhorn distance defined previously is asymmetrical, we need to calculate from both direction to remove this asymmetry. Then we run Sinkhorn algorithms over the transportation matrix generated by $N$ samples from $$Q(Z)$$ and $$P(Z)$$
respectively, and we are all set.

## Sinkhorn Algorithms for Sorting
