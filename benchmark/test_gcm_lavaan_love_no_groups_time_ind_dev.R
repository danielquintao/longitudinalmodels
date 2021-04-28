# GCM with time-independent deviations
# and w/o many explicit groups

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
dataset2 <- dataset[,1:4]
names(dataset2)[1] <- "y_i0"
names(dataset2)[2] <- "y_i1"
names(dataset2)[3] <- "y_i2"
names(dataset2)[4] <- "y_i3"

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
# deviations -> y_i
eps0 =~ 1.*y_i0
eps1 =~ 1.*y_i1
eps2 =~ 1.*y_i2
eps3 =~ 1.*y_i3
# variance is explained by deviations, and not by y_i
y_i0 ~~ 0*y_i0
y_i1 ~~ 0*y_i1
y_i2 ~~ 0*y_i2
y_i3 ~~ 0*y_i3
# deviations have 0 mean
eps0 ~ 0*1
eps1 ~ 0*1
eps2 ~ 0*1
eps3 ~ 0*1
# We are assuming that the deviations have the same variance
eps0 ~~ sigma_eps*eps0
eps1 ~~ sigma_eps*eps1
eps2 ~~ sigma_eps*eps2
eps3 ~~ sigma_eps*eps3
# fix lavaan\'s default of adding covariance to exogenous latent variables:
eta0 ~~ 0*eps0
eta0 ~~ 0*eps1
eta0 ~~ 0*eps2
eta0 ~~ 0*eps3
eta1 ~~ 0*eps0
eta1 ~~ 0*eps1
eta1 ~~ 0*eps2
eta1 ~~ 0*eps3
eps0 ~~ 0*eps1
eps0 ~~ 0*eps2
eps0 ~~ 0*eps3
eps1 ~~ 0*eps2
eps1 ~~ 0*eps3
eps2 ~~ 0*eps3
'
fit <- growth(model, data=dataset2)
summary(fit)
#                 Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.359    0.740  101.875    0.000
# eta1             -0.109    0.023   -4.749    0.000
#
# Variances:
# eps0    (sgm_)   31.284    3.010   10.392    0.000
# eps1    (sgm_)   31.284    3.010   10.392    0.000
# eps2    (sgm_)   31.284    3.010   10.392    0.000
# eps3    (sgm_)   31.284    3.010   10.392    0.000
# eta0             46.820    8.128    5.760    0.000
# eta1              0.022    0.008    2.558    0.011
#
# Covariances:
# eta0 ~~                                             
#   eta1              0.302    0.181    1.673    0.094
