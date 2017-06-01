"""
    Baseline
    \hat{r}_{ij} = \mu + b_u + b_i
    An estiamted rating is based on the average score and the bias of user average and item average.
"""
from scipy import sparse
import bl_fun as bl
import RecTool as rt


file_path = "input/ml-latest-small/"
file_name = file_path + "ratings.csv"
rate_m, test_data, user, item = rt.file_read(file_name)

test_hat = bl.pred(rate_m, test_data)

for i in range(len(test_data)):
    if i % 500 == 0:
        hat = test_hat[i]
        true = test_data[i]
        print("user:  "+str(hat[0])+"   movie:  "+str(hat[1]))
        print("true value -->  "+str(true[2])+" estimated value -->   "+str(hat[2]))

print(rt.loss_rmse(test_hat, test_data))
