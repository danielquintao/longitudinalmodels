# based on https://github.com/cran/flexmix/blob/master/R/models.R
library(flexmix)
ourBiasedModel <- function(formula=.~., offset=NULL)
{
  
  # family <- "gaussian"
  # glmrefit <- function(x, y, w) {
  #   fit <- c(glm.fit(x, y, weights=w, offset=offset,
  #                    family=get(family, mode="function")()),
  #            list(call = sys.call(), offset = offset,
  #                 control = eval(formals(glm.fit)$control),            
  #                 method = "weighted.glm.fit"))
  #   fit$df.null <- sum(w) + fit$df.null - fit$df.residual - fit$rank
  #   fit$df.residual <- sum(w) - fit$rank
  #   fit$x <- x
  #   fit
  # }
  
  z <- new("FLXMRglm", weighted=TRUE, formula=formula,
           name="ourBiasedModel", offset = offset)#,
           #family=family), refit=glmrefit)
  z@preproc.y <- function(x){
    if (ncol(x) > 1)
      stop(paste("family y must be univariate"))
    x
  }
  
  z@defineComponent <- function(para) {
    predict <- function(x, ...) {
      dotarg = list(...)
      if("offset" %in% names(dotarg)) offset <- dotarg$offset
      p <- x %*% para$coef
      if (!is.null(offset)) p <-  p + offset
      p
    }
      
    logLik <- function(x, y, ...)
      dnorm(y, mean=predict(x, ...), sd=para$sigma, log=TRUE)
      
    new("FLXcomponent",
        parameters=list(coef=para$coef, sigma=para$sigma),
        logLik=logLik, predict=predict,
        df=para$df)
  }
    
  z@fit <- function(x, y, w, component){
    fit <- lm.wfit(x, y, w=w, offset=offset)
    z@defineComponent(para = list(coef = coef(fit), df = ncol(x)+1,
                                  sigma =  sqrt(sum(fit$weights * fit$residuals^2 /
                                                      mean(fit$weights))/ (nrow(x))))) ### CHANGE IS HERE
  }
  
  z
}
