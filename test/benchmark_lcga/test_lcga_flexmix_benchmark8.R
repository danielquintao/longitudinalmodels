library(flexmix)
dataset <- read.csv("benchmark8_to_df.csv", header=TRUE)
m = stepFlexmix(observation ~ time|individual, data=dataset, k = 4, nrep=10,
                control=list(classify="weighted", iter.max=500))
parameters(m)
parameters(m)["sigma",]^2
prior(m)
posterior(m)
clusters(m) 

### flexmix with our biased model (based on original flexmix model)

# using initFlexMix is like using stepFlexmix, but we choose the tolerance
# of the two multistart stps
# (EXPLANATION: flexmix's multistart consists of running several
# short optimizations, and then taking the best one as starting point)
m2 = initFlexmix(observation ~ time|individual, data=dataset, k = 4, nrep=10,
                 control=list(classify="weighted", iter.max=500,
                              verbose=500, minprior=0),
                 model=ourBiasedModel(),
                 init = list(name = "tol.em",
                             step1 = list(tolerance = 1E-12), # no 'short' run
                             step2 = list()))

# or we can run flex mix many times and take the best result
# (i.e. traditional multistart)
# m2 = flexmix(observation ~ time|individual, data=dataset, k = 4,
#              control=list(classify="weighted", iter.max=500,
#                           verbose=500, minprior=0),
#              model=ourBiasedModel())

# or we can just run stepFlexMix
m2 = stepFlexmix(observation ~ time|individual, data=dataset, k = 4, nrep=10,
                 control=list(classify="weighted", iter.max=500,
                              verbose=0, minprior=0, tol=1E-12),
                 model=ourBiasedModel())

# parameters(m2)
round(parameters(m2), digits=5)
# parameters(m2)["sigma",]^2
round(parameters(m2)["sigma",]^2, digits=5)
prior(m2)
posterior(m2)
clusters(m2)        