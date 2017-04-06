"""
    Baseline
    \hat{r}_{ij} = \mu + b_u + b_i
    An estiamted rating is based on the average score and the bias of user average and item average.
"""
from scipy import sparse
import bl_fun as bl
import RecTool as rt


file_name = "ratings.csv"
rate_m, test_data = rt.file_read(file_name)

mu = bl.get_mu(rate_m)
bu, bi = bl.get_bu_bi(rate_m, mu)

test_hat = bl.pred(rate_m, test_data, mu, bu, bi)
print(rt.loss_rmse(test_hat, test_data))
