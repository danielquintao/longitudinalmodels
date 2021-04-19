### Test for Benchmark 1:
### 2 groups, NO covariance structure


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
# .eta0              1.546    0.209    7.397    0.000
# .eta1              0.343    0.289    1.189    0.235
# .eta2              0.501    0.230    2.174    0.030
# eta0 ~                                              
#   x                 2.412    0.275    8.786    0.000
# eta1 ~                                              
#   x                -0.515    0.379   -1.359    0.174
# eta2 ~                                              
#   x                 0.358    0.302    1.185    0.236
#
# Variances:
# .eta0             -0.055    0.373   -0.149    0.882
# .eta1             -5.491    1.882   -2.918    0.004
# .eta2             -1.157    1.238   -0.935    0.350
# eps0              1.148    0.437    2.624    0.009
# eps1              0.084    0.192    0.438    0.661
# eps2              0.564    0.356    1.584    0.113
# eps3              0.574    1.647    0.348    0.728
#
# Covariances:
# .eta0 ~~                                             
# .eta1              2.700    0.986    2.737    0.006
# .eta2             -0.353    0.481   -0.734    0.463
# .eta1 ~~                                             
# .eta2              4.310    1.076    4.007    0.000


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


