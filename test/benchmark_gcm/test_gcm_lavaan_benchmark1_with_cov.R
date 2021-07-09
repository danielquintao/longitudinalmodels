### Test for Benchmark 1:
### 2 groups, we keep the covariance structure


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
'
fit <- growth(model, data=dataset2)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# .eta0              1.546       NA
# .eta1              0.343       NA
# .eta2              0.501       NA
# eta0 ~                                              
#   x                2.412       NA  
# eta1 ~                                              
#   x               -0.515       NA   
# eta2 ~                                              
#   x                0.358       NA   
#
# Variances:
# .eta0              0.317       NA                  
# .eta1              0.427       NA                  
# .eta2              0.561       NA                  
#  eps0              0.775       NA                  
#  eps1              0.286       NA                  
#  eps2              0.703       NA                  
#  eps3              1.190       NA 
#
# Covariances:
# eps0 ~~                                             
# eps1              0.288       NA                  
# eps2              0.338       NA                  
# eps3             -0.221       NA                  
# eps1 ~~                                             
# eps2              0.133       NA                  
# eps3              0.079       NA                  
# eps2 ~~                                             
# eps3              0.358       NA                  
# .eta0 ~~                                             
# .eta1             0.769       NA                  
# .eta2             0.867       NA                  
# .eta1 ~~                                             
# .eta2             0.947       NA     


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
'
fit <- sem(semmodel, data=dataset2)
summary(fit)
# sam result as above


