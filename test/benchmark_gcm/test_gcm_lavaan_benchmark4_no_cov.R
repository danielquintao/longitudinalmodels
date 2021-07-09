### Test for Benchmark 4:
### 2 groups, uncorrelated deviations


library(lavaan)

dataset <- read.csv("playground_data/benchmark4_data.csv", header=FALSE)
temporal_measure <- dataset[,1:4]
names(temporal_measure)[1] <- "y_i0"
names(temporal_measure)[2] <- "y_i1"
names(temporal_measure)[3] <- "y_i2"
names(temporal_measure)[4] <- "y_i3"
groups <- dataset[5]
names(groups)[1] <- 'x'
dataset2 <- cbind(temporal_measure, groups)

# With growth function:

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
eta2 =~ 0.25*y_i1 + 1.*y_i2 + 2.25*y_i3
eta0 ~ x
eta1 ~ x
eta2 ~ x
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
eta2 ~~ 0*eps0
eta2 ~~ 0*eps1
eta2 ~~ 0*eps2
eta2 ~~ 0*eps3
eps0 ~~ 0*eps1
eps0 ~~ 0*eps2
eps0 ~~ 0*eps3
eps1 ~~ 0*eps2
eps1 ~~ 0*eps3
eps2 ~~ 0*eps3
'
fit <- growth(model, data=dataset2)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# .eta0              1.507    0.069   21.923    0.000
# .eta1              0.586    0.165    3.547    0.000
# .eta2              1.003    0.124    8.089    0.000
# eta0 ~                                              
#   x                 1.602    0.095   16.939    0.000
# eta1 ~                                              
#   x                 0.582    0.227    2.560    0.010
# eta2 ~                                              
#   x                 0.619    0.171    3.625    0.000
#
# Variances:
# eps0               -0.397    0.185   -2.145    0.032
# eps1                1.502    0.175    8.568    0.000
# eps2               -0.322    0.196   -1.640    0.101
# eps3                4.599    1.024    4.491    0.000
# .eta0               0.965    0.201    4.796    0.000
# .eta1               0.066    0.735    0.089    0.929
# .eta2              -3.552    0.890   -3.990    0.000
#
# Covariances:
# .eta0 ~~                                             
#   .eta1            -0.333    0.365   -0.914    0.361
#   .eta2             0.821    0.196    4.189    0.000
# .eta1 ~~                                             
#   .eta2             3.829    0.643    5.953    0.000


