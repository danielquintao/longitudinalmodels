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
'
fit <- growth(model, data=dataset)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0              0.900       NA
# eta1              0.589       NA 
#
# Variances:
# eta0              0.049       NA                  
# eta1              0.460       NA                  
# eps0              0.244       NA                  
# eps1              0.354       NA                  
# eps2              0.840       NA                  
# eps3              0.873       NA   
#
# Covariances:
#   eta0 ~~                                             
# eta1              0.595       NA                  
# eps0 ~~                                             
#   eps1            0.095       NA                  
#   eps2           -0.049       NA                  
#   eps3           -0.178       NA                  
# eps1 ~~                                             
#   eps2            0.397       NA                  
#   eps3            0.277       NA                  
# eps2 ~~                                             
#   eps3            0.588       NA                  


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
'
fit <- sem(semmodel, data=dataset)
summary(fit)
# sam result as above


