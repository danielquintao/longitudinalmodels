# longitudinalmodels

This project aims at  providing trustworthy and easy-to-use code for the estimation of different longitudinal models.

In the folder ```extra``` we find the implementation of some pipelines that deals with most of the boring part for applying or models, and also manages data to be outputed.

The folder ```models``` contains the imlementations of our models. There are:
- GCM (Growth Curve Model)
- Extended GCM  (GCM with a covariable encoding kown groups to which the subjetcs belong). The groups are **not** time-dependent (i.e. each subject belongs to the same group throughout all the study)
- GCM (Growth Curve Model)
- LCGA (Latent Class Growth Analysis), which is a mixture of regressions. It combines clustering and linear regression

The folder ```test``` contains sample data and testing code to evaluate our models. This is not a traditional "test folder" as we encounter in many software repositories to allow for safe pull requests

The folder ```utils``` has auxiliary methods for the models and pipelines, in particular for plotting
