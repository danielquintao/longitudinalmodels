### Combination of test_gcm_lavaan_2 and test_gcm_lavaan_3 w/ Benchmark 1:
### We have 3 groups encoded w/ 2 binary vars AND we add a covariance structure
### to the manifest variables y


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

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
eta2 =~ 0.25*y_i1 + 1.*y_i2 + 2.25*y_i3
eta0 ~ x
eta1 ~ x
eta2 ~ x
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
# eta0             1.546       NA
# eta1             0.343       NA 
# eta2             0.501       NA 
# Regressions:
# eta0 ~ x         2.412       NA   
# eta1 ~ x        -0.515      NA
# eta2 ~ x         0.358      NA
