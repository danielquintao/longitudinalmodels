### Test for Benchmark 2:
### NO groups, NO covariance structure


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
# eta0              0.894    0.079   11.255    0.000
# eta1              0.518    0.156    3.324    0.001
#
# Variances:
# eps0              0.096    0.033    2.926    0.003
# eps1              0.066    0.023    2.904    0.004
# eps2              0.202    0.060    3.396    0.001
# eps3              0.386    0.123    3.149    0.002
# eta0              0.247    0.065    3.773    0.000
# eta1              1.060    0.245    4.321    0.000
# eta0~~eta1        0.441    0.104    4.255    0.000

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


