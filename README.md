# ertransfer

`ertransfer` is a Python package consisting of recent **transfer learning 
algorithms for entity resolution**.

Entity Resolution (ER) is the process of linking records of the same entity 
across one or more databases in the absence of unique entity identifiers. 
Supervised methods for ER, however, often require extensive human efforts to 
label record pairs as matches and non-matches. An intuitive approach to avoid
this tedious labelling task is to utilise existing labelled training data 
from a related domain. The idea of utilising a source domain with a different 
distribution to label a target domain has attracted the attention of researchers 
and given rise to the area of transfer learning (TL). 


## Contents and usage

The `ertransfer` package contains the implementations of the following state 
of the art tranfer learning algorithms for entity resolution:

1. The **DTAL_star** algorithm is an deep transfer learning algorithm for ER 
that performs data set adaptation via gradient reversal. However, in this 
package we have not implemented the active learning strategy proposed in the 
original paper. [[1](https://arxiv.org/pdf/1906.08042.pdf)].

Given a source domain {**Xs**, *Ys*} and a target domain {**Xt**, *Yt*}, the
 algorithm is applied as follows:
  
```python
from ertransfer.models import DTAL_star
import pandas as pd

# Load data sets
Xs = pd.read_csv('data/dblp-acm/sampled-0.66-0-A-A.csv')
Xt = pd.read_csv('data/dblp-scholar/sampled-0.66-0-A-A.csv')

# train and predict
dtal_baseline = DTAL_star()
dtal_baseline.fit(Xs, Xt, Xs_name='dblp-acm', Xt_name='dblp-scholar')

```

## Package structure
| Directory | Contains.. |
|---------------------|--------------------------------------------------------|
| data/               | Data sets 
| deepmatcher/        | Updated deep matcher library to use with DTAL |
| ertransfer/models/  | Transfer learning algorithms for ER           |
| ertransfer/samples/ | Sample codes to use the algorithms           |


## Dependencies

The `ertransfer` package requires the following python packages to be installed:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Pandas](http://www.scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Torch](https://pytorch.org/)


## Contact

Contact the author of the package: [nishadi.kirielle@anu.edu.au](mailto:nishadi.kirielle@anu.edu.au)


## References

[1] Jungo Kasai, Kun Qian, Sairam Gurajada, Yunyao Li, and Lucian Popa. (2019) 
*Low-resource Deep Entity Resolution with Transfer and Active Learning.* 
In Annual Meeting of the Association for Computational Linguistics, ACL, 
Florence.
