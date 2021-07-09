### Test for Benchmark 2:
### NO groups, time-independent deviations

# note that time-independent deviations do not mean equal errors for the same
# individual across time-points (which would be a huge coincidence), but
# rather that the T errors have the same variance


library(lavaan)

dataset <- read.csv("playground_data/benchmark2_data.csv", header=FALSE)
names(dataset)[1] <- "y_i0"
names(dataset)[2] <- "y_i1"
names(dataset)[3] <- "y_i2"
names(dataset)[4] <- "y_i3"

# With growth function:

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
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
fit <- growth(model, data=dataset)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0              0.872    0.081   10.764    0.000
# eta1              0.558    0.149    3.738    0.000
#
# Variances:
# eps0    (sgm_)    0.175    0.025    7.071    0.000
# eps1    (sgm_)    0.175    0.025    7.071    0.000
# eps2    (sgm_)    0.175    0.025    7.071    0.000
# eps3    (sgm_)    0.175    0.025    7.071    0.000
# eta0              0.206    0.068    3.033    0.002
# eta1              0.974    0.224    4.356    0.000
# eta0~~eta1        0.458    0.100    4.572    0.000

#Warning message:
# In lav_object_post_check(object) :
#   lavaan WARNING: covariance matrix of latent variables
# is not positive definite;
# use lavInspect(fit, "cov.lv") to investigate.

#indeed NOTE that
# corr(eta0,eta1) = cov(eta0,eta1)/(sqrt(var(eta(0)))*sqrt(var(eta(0)))) =
# = 1.022474 > 1




# With sem function
semmodel <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
# compute intercept and slope for eta
eta0 ~ 1
eta1 ~ 1
# covariance in eta
# eta0 ~~ eta0
# eta1 ~~ eta1
# eta0 ~~ eta1
# deviations -> y_i
eps0 =~ 1.*y_i0
eps1 =~ 1.*y_i1
eps2 =~ 1.*y_i2
eps3 =~ 1.*y_i3
# variance is explained by deviations, and not by y_i
# eps0 ~~ eps0
# eps1 ~~ eps1
# eps2 ~~ eps2
# eps3 ~~ eps3
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
eps0 ~~ 0*eps1
eps0 ~~ 0*eps2
eps0 ~~ 0*eps3
eps1 ~~ 0*eps2
eps1 ~~ 0*eps3
eps2 ~~ 0*eps3
'
fit <- sem(semmodel, data=dataset)
summary(fit)
# sam result as above


