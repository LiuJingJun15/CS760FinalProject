# CS760FinalProject

This is the final project for CS760: Machine Learning
Usage:

Linear Regression Method:
make plot with state data: linear_reg_plot_state()
10-fold cross validation with state data: accu, dev = linear_reg_cv_state()
make plot with county data: linear_reg_plot_county()
5-fold cross validation with county data: accu, dev = linear_reg_cv_county()


Distance-weighted Nearest Neighbour Method:
make plot with state data: KNN_plot_state()
10-fold cross validation with state data: accu, dev = KNN_cv_state()
10-fold cross validation for n times with state data: accu, dev = KNN_cv_state_n(n)
make plot with county data: KNN_plot_county()
5-fold cross validation with county data: accu, dev = KNN_validate_county()
5-fold cross validation for n times with county data: accu, dev = KNN_validate_county_ntimes(n)
