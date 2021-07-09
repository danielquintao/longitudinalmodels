library(flexmix)
data(NPreg)
m1 = flexmix(yn ~ x + I(x^2), data = NPreg, k = 2)
m1
parameters(m1)
# parameters(m1, component = 2) # too see a specific class
prior(m1)
posterior(m1) # like our method get_clusterwise_probabilities
clusters(m1) # like our method get_predictions

# run 10 times and get best results
m = stepFlexmix(yn ~ x + I(x^2), data = NPreg, k = 2, nrep=10)
# control = list(classify="auto") or control = list(classify="weighted")?
m
