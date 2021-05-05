### Test for Benchmark 4:
### 2 groups, time-independent deviations

# note that time-independent deviations do not mean equal errors for the same
# individual across time-points (which would be a huge coincidence), but
# rather that the T errors have the same variance


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
# .eta0              1.486    0.074   20.117    0.000
# .eta1              0.540    0.176    3.073    0.002
# .eta2              1.012    0.125    8.125    0.000
# eta0 ~                                              
#   x                 1.589    0.102   15.634    0.000
# eta1 ~                                              
#   x                 0.554    0.242    2.289    0.022
# eta2 ~                                              
#   x                 0.624    0.171    3.641    0.000
#
# Variances:
# eps0    (sgm_)    0.741    0.066   11.180    0.000
# eps1    (sgm_)    0.741    0.066   11.180    0.000
# eps2    (sgm_)    0.741    0.066   11.180    0.000
# eps3    (sgm_)    0.741    0.066   11.180    0.000
# .eta0             -0.060    0.085   -0.704    0.482
# .eta1             -3.620    0.727   -4.981    0.000
# .eta2             -1.135    0.312   -3.643    0.000
#
# Covariances:
# .eta0 ~~                                             
#   .eta1              2.182    0.174   12.528    0.000
# .eta2             -0.341    0.099   -3.457    0.001
# .eta1 ~~                                             
#   .eta2              3.566    0.434    8.225    0.000


