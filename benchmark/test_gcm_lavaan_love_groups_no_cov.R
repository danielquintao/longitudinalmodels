### This file computes the intercept and slope for the love dataset, but assuming
### that we have the groups (0,0), (1,0), (0,1) encoded by 2 binary variables!
### This is GCM with known groups 
### moreover, we assumed NO covariance between different deviations

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
# x1               -4.082    1.740   -2.345    0.019
# x2                1.170    1.766    0.662    0.508
# eta1 ~                                              
# x1               -0.117    0.052   -2.226    0.026
# x2                0.044    0.053    0.826    0.409
#.eta0             76.419    1.214   62.955    0.000
#.eta1             -0.089    0.037   -2.448    0.014
#
# Variances:
#.eta0             45.331    7.734    5.861    0.000
#.eta1              0.001    0.016    0.062    0.951
# eps0             24.342    5.385    4.520    0.000
# eps1             26.842    4.990    5.379    0.000
# eps2             37.158    6.291    5.906    0.000
# eps3             54.943   20.153    2.726    0.006
# Covariances:
#.eta0 ~~                                             
# .eta1            0.057    0.177    0.320    0.749


