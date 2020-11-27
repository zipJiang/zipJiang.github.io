---
layout: posts
title:  "Sinkhorn's Theorem, Sinkhorn Alorithm and Applications"
---

# Sinkhorn's Theorem, Sinkhorn Algorithm and Applications

In this post I'm going to briefly discuss a computationally efficient way to calculate an approximate matrix factorization, called Sinkhorn's algorithm, and it's example application in machine learning.

### Sinkhorn's Theorem and Sinkhorn-Knopp Algorithm

The Sinkhorn's theorem states that every square matrix with strictly positive elements can be transformed into a doubly stochastic matrix $D_1AD_2$. Where $D_1$ and $D_2$ are diagonal matrices with all positive main diagonals. The matrices $D_1$ and $D_2$ are themselves unique up to a constant factor.

Sounds simple, except the problem that what is a doubly stochastic matrix? Well, a doubly stochastic matrix is a all non-negative square matrix that is both row-normalized and column-normalized. To speak about this mathematically for a non-negative square matrix $A \in \mathcal{R}^{N \times N}$, $\sum_{i = 1}^{N} a_{ij} = 1, \forall j \in [1, N]$.
