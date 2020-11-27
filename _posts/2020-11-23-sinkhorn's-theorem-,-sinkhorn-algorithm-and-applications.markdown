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

## Optimal Transport

We are now considering a very interesting application of this algorithm, which is in calculating optimal transport distance. Optimal transport is useful as a distance metric between distributions, in which way it plays a very important role in GAN and \*AE to stablize training.

So how is calculating a doubly stochastic matrix related to optimal transport?
