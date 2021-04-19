# GCM with NO covariance between different deviations,
# and w/o many explicit groups

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
dataset2 <- dataset[,1:4]
names(dataset2)[1] <- "y_i0"
names(dataset2)[2] <- "y_i1"
names(dataset2)[3] <- "y_i2"
names(dataset2)[4] <- "y_i3"

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
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
fit <- growth(model, data=dataset2)
summary(fit)
#                 Estimate  Std.Err  z-value  P(>|z|)
# eta0             75.175       NA 
# eta1             -0.115       NA 
#
# Variances:
# eta0             31.085       NA                  
# eta1             -0.054       NA                  
# eps0             55.956       NA                  
# eps1             42.578       NA                  
# eps2             30.346       NA                  
# eps3            106.538       NA  
#
# Covariances:
# eta0 ~~                                             
# eta1              0.950       NA                  
# eps0 ~~                                             
# eps1             24.867       NA                  
# eps2              8.189       NA                  
# eps3             -3.365       NA                  
# eps1 ~~                                             
# eps2             -0.473       NA                  
# eps3              2.306       NA                  
# eps2 ~~                                             
# eps3             -0.469       NA  
