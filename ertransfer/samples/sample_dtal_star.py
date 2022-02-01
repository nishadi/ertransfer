import logging
import time
import pandas as pd
from ertransfer.models import DTAL_star

if __name__ == '__main__':
  start_t = time.time()

  # Initialize logging
  log_file_name = 'dtal*.log'
  logging.basicConfig(filename=log_file_name, level=logging.INFO,
                      format='%(asctime)s %(message)s', filemode='w')


  Xs = pd.read_csv('data/dblp-acm/sampled-0.66-0-A-A.csv')
  Xt = pd.read_csv('data/dblp-scholar/sampled-0.66-0-A-A.csv')

  dtal_baseline = DTAL_star()
  dtal_baseline.fit(Xs, Xt, Xs_name='dblp-acm', Xt_name='dblp-scholar')

  logging.info(
    'Program successfully finished {} s!'.format(time.time() - start_t))
