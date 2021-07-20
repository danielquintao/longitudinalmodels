library(flexmix)
dataset <- read.csv("benchmark7_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ time|individual, data=dataset, k = 3, nrep=10,
                control=list(classify="weighted"))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

### flexmix with our biased model (based on original flexmix model)
m2 = stepFlexmix(observation ~ time|individual, data=dataset, k = 3, nrep=10,
                 control=list(classify="weighted", iter.max=500,
                              verbose=0, minprior=0, tol=1E-10),
                 model=ourBiasedModel())
# parameters(m2)
round(parameters(m2), digits=5)
# parameters(m2)["sigma",]^2
round(parameters(m2)["sigma",]^2, digits=5)
prior(m2)
posterior(m2)
clusters(m2)        
