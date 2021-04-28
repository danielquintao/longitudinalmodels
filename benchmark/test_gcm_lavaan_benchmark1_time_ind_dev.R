### Test for Benchmark 1:
### 2 groups, time-independent deviations

# note that time-independent deviations do not mean equal errors for the same
# individual across time-points (which would be a huge coincidence), but
# rather that the T errors have the same variance


library(lavaan)

dataset <- read.csv("playground_data/benchmark1_data.csv", header=FALSE)
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
# .eta0              1.551    0.218    7.125    0.000
# .eta1              0.347    0.293    1.184    0.237
# .eta2              0.495    0.240    2.063    0.039
# eta0 ~                                              
#   x                 2.363    0.286    8.267    0.000
# eta1 ~                                              
#   x                -0.557    0.385   -1.446    0.148
# eta2 ~                                              
#   x                 0.413    0.315    1.309    0.190
#
# Variances:
# .eta0              0.637    0.212    3.008    0.003
# .eta1             -1.895    0.824   -2.301    0.021
# .eta2             -0.300    0.387   -0.774    0.439
# eps0    (sgm_)    0.378    0.076    5.000    0.000
# eps1    (sgm_)    0.378    0.076    5.000    0.000
# eps2    (sgm_)    0.378    0.076    5.000    0.000
# eps3    (sgm_)    0.378    0.076    5.000    0.000
#
# Covariances:
# .eta0 ~~                                             
#   .eta1              1.049    0.250    4.200    0.000
#   .eta2              0.461    0.209    2.201    0.028
# .eta1 ~~                                             
#   .eta2              2.582    0.501    5.153    0.000


# With sem function
semmodel <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
eta2 =~ 0.25*y_i1 + 1.*y_i2 + 2.25*y_i3
eta0 ~ x
eta1 ~ x
eta2 ~ x
# compute intercept and slope for eta
eta0 ~ 1
eta1 ~ 1
eta2 ~ 1
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
# observations and deviations have 0 mean
eps0 ~ 0*1
eps1 ~ 0*1
eps2 ~ 0*1
eps3 ~ 0*1
y_i0 ~ 0*1
y_i1 ~ 0*1
y_i2 ~ 0*1
y_i3 ~ 0*1
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
fit <- sem(semmodel, data=dataset2)
summary(fit)
# sam result as above


