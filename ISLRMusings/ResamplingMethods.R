# Resampling methods applied to chess dataset
library(boot)
games = read.csv("games_new_vars.csv")


##### Validation set approach 

set.seed(1)
split = sample(nrow(games), 1) / nrow(games)
split = round(nrow(games) * split)

train.indices = sample(nrow(games), size = split)
train.data = games[train.indices,]
test.data = games[-train.indices,]

nrow(train.data) + nrow(test.data)

##### Now we can train a model on the training data and test it on the test data

###############################################################################################

##### Leave-one-out-cross validation 

##### Many models include a built in LOOCV option. Here, we show how to perform LOOCV
##### on a logistic regression model. 

## Logistic regression example 
glm.fit = glm(higher_rating_won ~ abs_diff_rating,family = binomial(link='logit'), data = games)
# cv.error = cv.glm(games, glm.fit)
# This takes a very long time to run. If this finishes running, you look at the delta attribute of cv.error. 
# The first number will be the training MSE (for classification problems, ERR), and 
# the second will be the test ERR.

###############################################################################################

###### K-fold cross validation 

###### Many models also include their built in K-fold option. Here, we show how to perform 
###### k-fold cross validation on our previous logistic regression model. 

cv.error = cv.glm(games, glm.fit, K = 5)
cv.error$delta

cv.error = cv.glm(games, glm.fit, K = 10)
cv.error$delta

###############################################################################################

####### Bootstrapping 

# We will calculate bootstrap the mean of the absolute value differences in rating 
meanFunc <- function(x,i){mean(x[i])}
boot(games$abs_diff_rating, meanFunc, R=1000)
# An interpretation for this call: original mean is given by "original", and this 
# carries a standard error of 1.2517. 

###############################################################################################

