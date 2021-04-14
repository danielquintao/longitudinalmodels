### Combination of test_gcm_lavaan_2 and test_gcm_lavaan_3:
### We have 3 groups encoded w/ 2 binary vars AND we add a covariance structure
### to the manifest variables y


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

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ -3.*y_i0 + 3.*y_i1 + 9.*y_i2 + 36.*y_i3
eta0 ~ x1 + x2
eta1 ~ x1 + x2
y_i0 ~~ y_i1
y_i0 ~~ y_i2
y_i0 ~~ y_i3
y_i1 ~~ y_i2
y_i1 ~~ y_i3
y_i2 ~~ y_i3
'

fit <- growth(model, data=dataset2)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0            76.468       NA
# eta1            -0.090       NA 
# Regressions:
# eta0 ~ x1       -4.278       NA   
# eta0 ~ x2        0.480       NA
# eta1 ~ x1       -0.116       NA
# eta1 ~ x2        0.047       NA  
