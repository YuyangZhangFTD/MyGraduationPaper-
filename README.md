# MyGraduationPaper-
The code about my graduation paper in recommendation system.

The data is in 'input/' and for convenience, I use the movielens data.

The data is too large to upload, so, if you want to download the data, [click here](https://grouplens.org/datasets/movielens/) or email me: zhangyuyang4d@163.com

## My Paper
When I study the deep learning in NLP, I found word2vec can be used in recommendation system.
And according to [1,2,3], I try to model with MovieLens data. 
However... when I read [4], I find this has been done...Orz.
Luckily, the details of my model are different. 
As a undergraduate student, I try my best...

## Details
I call this model as Item2Vec++(Item to Vector plus)...XDDDD.

And the details of model can be viewed in [4.1item2vec+.pdf](https://github.com/YuyangZhangFTD/MyGraduationPaper-/blob/master/4.1Item2Vec%2B.pdf).

As for the code, the cf represents the collaborative filter, the mf represents the matrix factorization and the bl represents the baseline.
Run **_main.py to test, and **_fun.py contain the functions of ** method.


## Reference
[1] Mikolov T, Chen K, Corrado G, et al. Efficient estimation of word representations in vector
space[J]. arXiv preprint arXiv:1301.3781, 2013.

[2] Rong X. word2vec parameter learning explained[J]. arXiv preprint arXiv:1411.2738, 2014.

[3] Bengio Y, Ducharme R, Vincent P, et al. A neural probabilistic language model[J]. Journal
of machine learning research, 2003, 3(Feb): 1137-1155.

[4] Barkan O, Koenigstein N. Item2vec: neural item embedding for collaborative
filtering[C]//Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International
Workshop on. IEEE, 2016: 1-6.
