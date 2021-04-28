### This file computes the intercept and slope for the love dataset, but assuming
### that we have the groups (0,0), (1,0), (0,1) encoded by 2 binary variables!
### This is GCM with known groups 
### moreover, we assumed time-independent deviations

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
temporal_measure <- dataset[,1:4]
names(temporal_measure)[1] <- "y_i0"
names(temporal_measure)[2] <- "y_i1"
names(temporal_measure)[3] <- "y_i2"
names(temporal_measure)[4] <- "y_i3"
groups <- dataset[6:7]
names(groups)[1] <- 'x1'
names(groups)[2] <- 'x2'
dataset2 <- cbind(temporal_measure, groups)

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
eta0 ~ x1 + x2
eta1 ~ x1 + x2
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
# eta0 ~                                              
#   x1               -4.148    1.713   -2.422    0.015
# x2                0.953    1.739    0.548    0.584
# eta1 ~                                              
#   x1               -0.114    0.053   -2.146    0.032
# x2                0.057    0.054    1.064    0.287
# .eta0             76.442    1.195   63.979    0.000
# .eta1             -0.089    0.037   -2.410    0.016
#
# # Variances:
# eps0    (sgm_)   31.284    3.010   10.392    0.000
# eps1    (sgm_)   31.284    3.010   10.392    0.000
# eps2    (sgm_)   31.284    3.010   10.392    0.000
# eps3    (sgm_)   31.284    3.010   10.392    0.000
# .eta0             41.971    7.476    5.614    0.000
# .eta1              0.017    0.008    2.127    0.033
#
# Covariances:
# .eta0 ~~                                             
#   .eta1              0.149    0.168    0.891    0.373


