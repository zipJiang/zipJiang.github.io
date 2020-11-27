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

It's obviousv that the algorithm has a complexity of $$O(N^2)$$ if applied a constant times at every step. Also notice that as we could write normalization as matrix multiplication, Sinkhorn algorithm is fully differentiable.

## Optimal Transport

We are now considering a very interesting application of this algorithm, which is in calculating optimal transport distance. Optimal transport is useful as a distance metric between distributions, in which way it plays a very important role in GAN and \*AE to stablize training.

So how is calculating a doubly stochastic matrix related to optimal transport?

First notice that optimal transport problem could be expressed in matrix form. Suppose we have $$N$$ number of starting points and $$N$$ number of destinations. Also, we have a distance matrix $$M \in \mathcal{R}^{N \times N}$$ where entry $$m_{ij}$$ indicates the distance, or in another word, the cost of moving items from starting point $$i$$ to destination $$j$$.

Now suppose our cargos are randomly distributed according to $$r$$ among the $$N$$ starting points. And we would like them to be distributed according to another distribution $$c$$. Now we can write up a matrix $$A \in \mathcal{R}^{N \times N}$$, where $$a_{ij}$$ corresponds to how much portion of the total cargos we would like to transport from starting point $$i$$ to destination $$j$$. Another way to view this matrix $$A$$ is that $$A$$ defines a joint distribution in the
transportation space and $$r$$ and $$c$$ could be seen as their marginals of starting points and destinations respectively. Now it's not hard to see that the final transportation cost would be the element-wise product between $$A$$ and $$M$$, or in other words the Frobenius inner product $$\langle A,M \rangle$$.
