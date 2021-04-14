### Test for Benchmark 2:
### NO groups
### We add a covariance structure to the manifest variables y


library(lavaan)

dataset <- read.csv("playground_data/benchmark2_data.csv", header=FALSE)
names(dataset)[1] <- "y_i0"
names(dataset)[2] <- "y_i1"
names(dataset)[3] <- "y_i2"
names(dataset)[4] <- "y_i3"

# We'll not write the covariances etc explicitly, because:
#
# "Technically, the growth() function is almost identical to the sem() function. 
# But a mean structure is automatically assumed, and the observed intercepts are
# fixed to zero by default, while the latent variable intercepts/means are 
# freely estimated." -- https://lavaan.ugent.be/tutorial/growth.html

model <- '
eta0 =~ 1.*y_i0 + 1.*y_i1 + 1.*y_i2 + 1.*y_i3
eta1 =~ 0.5*y_i1 + 1.*y_i2 + 1.5*y_i3
y_i0 ~~ y_i1
y_i0 ~~ y_i2
y_i0 ~~ y_i3
y_i1 ~~ y_i2
y_i1 ~~ y_i3
y_i2 ~~ y_i3
'
fit <- growth(model, data=dataset)
summary(fit)
#                Estimate  Std.Err  z-value  P(>|z|)
# eta0             0.900       NA
# eta1             0.589       NA 

