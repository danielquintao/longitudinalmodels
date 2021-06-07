### This file computes the intercept and slope for the love dataset, but assuming
### that we have the groups (0,0), (1,0), (0,1) encoded by 2 binary variables!
### This is GCM with known groups 

library(lavaan)

dataset <- read.csv("playground_data/lovedata.csv")
temporal_measure <- dataset[,1:4]
names(temporal_measure)[1] <- "y_i0"
names(temporal_measure)[2] <- "y_i1"
names(temporal_measure)[3] <- "y_i2"
names(temporal_measure)[4] <- "y_i3"
groups <- dataset[6:7]
names(groups)[1] <- 'x1'
names(groups)[2] <- 'x2'
dataset2 <- cbind(temporal_measure, groups)

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
eta0 ~ x1 + x2
eta1 ~ x1 + x2
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
# eta0 ~                                              
#  x1               -4.278       NA                  
#  x2                0.480       NA                  
# eta1 ~                                              
#  x1               -0.116       NA                  
#  x2                0.047       NA 
# .eta0             76.468       NA                  
# .eta1             -0.090       NA   
#
# Variances:
# .eta0             27.911       NA                  
# .eta1             -0.051       NA                  
#  eps0             53.082       NA                  
#  eps1             41.670       NA                  
#  eps2             30.464       NA                  
#  eps3             98.934       NA 
# Covariances:
# eps0 ~~                                             
# eps1             23.010       NA                  
# eps2              6.890       NA                  
# eps3             -3.100       NA                  
# eps1 ~~                                             
# eps2             -0.862       NA                  
# eps3              2.523       NA                  
# eps2 ~~                                             
# eps3             -0.579       NA                  
#.eta0 ~~                                             
#.eta1              0.731       NA 


