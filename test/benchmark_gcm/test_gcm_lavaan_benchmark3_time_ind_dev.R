### Test for Benchmark 3:
### NO groups, time-independent deviations

# note that time-independent deviations do not mean equal errors for the same
# individual across time-points (which would be a huge coincidence), but
# rather that the T errors have the same variance


library(lavaan)

dataset <- read.csv("playground_data/benchmark3_data.csv", header=FALSE)
names(dataset)[1] <- "y_i0"
names(dataset)[2] <- "y_i1"
names(dataset)[3] <- "y_i2"
names(dataset)[4] <- "y_i3"
names(dataset)[5] <- "y_i4"

# With growth function:

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3 + 1.*y_i4
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3 + 2.*y_i4
eta2 =~ 0.25*y_i1 + 1.*y_i2 + 2.25*y_i3 + 4.*y_i4
# deviations -> y_i
eps0 =~ 1.*y_i0
eps1 =~ 1.*y_i1
eps2 =~ 1.*y_i2
eps3 =~ 1.*y_i3
eps4 =~ 1.*y_i4
# variance is explained by deviations, and not by y_i
y_i0 ~~ 0*y_i0
y_i1 ~~ 0*y_i1
y_i2 ~~ 0*y_i2
y_i3 ~~ 0*y_i3
y_i4 ~~ 0*y_i4
# deviations have 0 mean
eps0 ~ 0*1
eps1 ~ 0*1
eps2 ~ 0*1
eps3 ~ 0*1
eps4 ~ 0*1
# We are assuming that the deviations have the same variance
eps0 ~~ sigma_eps*eps0
eps1 ~~ sigma_eps*eps1
eps2 ~~ sigma_eps*eps2
eps3 ~~ sigma_eps*eps3
eps4 ~~ sigma_eps*eps4
# fix lavaan\'s default of adding covariance to exogenous latent variables:
eta0 ~~ 0*eps0
eta0 ~~ 0*eps1
eta0 ~~ 0*eps2
eta0 ~~ 0*eps3
eta0 ~~ 0*eps4
eta1 ~~ 0*eps0
eta1 ~~ 0*eps1
eta1 ~~ 0*eps2
eta1 ~~ 0*eps3
eta1 ~~ 0*eps4
eta2 ~~ 0*eps0
eta2 ~~ 0*eps1
eta2 ~~ 0*eps2
eta2 ~~ 0*eps3
eta2 ~~ 0*eps4
eps0 ~~ 0*eps1
eps0 ~~ 0*eps2
eps0 ~~ 0*eps3
eps0 ~~ 0*eps4
eps1 ~~ 0*eps2
eps1 ~~ 0*eps3
eps1 ~~ 0*eps4
eps2 ~~ 0*eps3
eps2 ~~ 0*eps4
eps3 ~~ 0*eps4
'
fit <- growth(model, data=dataset)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0              0.289    0.093    3.094    0.002
# eta1              0.074    0.150    0.493    0.622
# eta2              0.775    0.079    9.766    0.000
#
# Variances:
# eps0    (sgm_)    0.450    0.028   15.811    0.000
# eps1    (sgm_)    0.450    0.028   15.811    0.000
# eps2    (sgm_)    0.450    0.028   15.811    0.000
# eps3    (sgm_)    0.450    0.028   15.811    0.000
# eps4    (sgm_)    0.450    0.028   15.811    0.000
# eta0              1.784    0.197    9.065    0.000
# eta1              3.372    0.521    6.470    0.000
# eta2              1.062    0.145    7.342    0.000
# 
# Covariances:
# eta0 ~~                                             
#   eta1             -0.336    0.235   -1.431    0.152
#   eta2              1.057    0.145    7.306    0.000
# eta1 ~~                                             
#   eta2             -1.187    0.243   -4.878    0.000




