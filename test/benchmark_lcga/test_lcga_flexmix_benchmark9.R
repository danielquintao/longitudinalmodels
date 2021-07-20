library(flexmix)
dataset <- read.csv("benchmark9_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ (time+I(time^2)|individual), data=dataset, k = 3, nrep=10,
                control=list(classify="weighted", iter.max=500))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

### flexmix with our biased model (based on original flexmix model)
m2 = stepFlexmix(observation ~ (time+I(time^2)|individual), data=dataset, k = 3,
                 nrep=10,
                 control=list(classify="weighted", iter.max=500,
                              verbose=0, minprior=0, tol=1E-10),
                 model=ourBiasedModel())
# parameters(m2)
round(parameters(m2), digits=3)
# parameters(m2)["sigma",]^2
round(parameters(m2)["sigma",]^2, digits=3)
prior(m2)
posterior(m2)
clusters(m2)        