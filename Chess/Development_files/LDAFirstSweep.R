library(MASS)

games <- read.csv('games_new_vars.csv')
attach(games)
names(games)

ld.fit <- lda(result ~ abs_diff_rating, data = games)

names(ld.fit)

summary(ld.fit)

ld.fit

Our prior probabilities correspond to the proportion of observations that 
fall in each class. Our probabilities should all add up to one.

ld.fit$prior[1] + ld.fit$prior[2] + ld.fit$prior[3]

Which they do. 

Our group means suggest, interestingly enough, that when we see a loss in the higher rated
player, on average the absolute difference in rating is 116.0524. Likewise, when draws
occur, the average absolute difference in rating is 147.6284. Finally, if we see a win 
from the higher rated player, we expect the absolute difference in rating to be on 
average 206.2477. 

These results turn out surprising. I would expect a small absolute difference in rating to
have more draws. These results are also a big sham since I trained the model on the entire
set. 

Now, we will check out the predictions. This is experimental so we will do a simple 
cross set validation with a split of 70 / 30 for the data. For this, we will 
refit on training before testing as well. 

n = nrow(games)
shuffle.data  games[sample(n), ]
train.data = shuffle.data[1:round(.7*n), ]
test.data = shuffle.data[(round(.7*n)+1):n, ]

lda.fit <- lda(result ~ abs_diff_rating, data = train.data)
lda.fit

predictions <- predict(lda.fit, test.data)
names(predictions)

table(predictions$class)

We predicted a win for the higher rating for everyone it seems. 
Uh oh. Big yike houston.

It could be that our test.data really did have a win win for everyone. 

plot(lda.fit)


We can manually find the error rate.

mistakes <- 2 - test.data$result
sum(mistakes) / nrow(test.data)

The error rate for this model is stunningly awful. It could be that our threshold for probabilities is 
too lenient. Let us see if fixing that gives us hope. 

Looking at the posterior probabilities clarifies the workings of the model for us. 

sum(predictions$posterior[,3] > .7)

You would think that there would be an easier way to fix the posterior probabilities. 

lda.fit

# Although, could the predictions be better if we upped the threscholds manually? 

sum(predictions$posterior[,1] > .5)

# Insights! The model gives every observation at least a .5 chance of winning since it 
# the higher rating should assume that in practice ish nish. Ideas...

# Arbitrarily deciding to give a threschold of 65 percent if someone were to win.  

modified.preds <- predictions$posterior

modified.preds[modified.preds[,3] < 0.60,3] = 0

# Now we can assign classes manually here. The percentage of a draw is extremely low so we will 
# simply assign a zero in the case that two is zero. 

results <- c(1:6017)
length(results)
modified.preds <- rbind(modified.preds, results)

mod.pred.results <- c()
for(i in 1:length(test.data$result)){
  if(modified.preds[i,3] == 0){
    mod.pred.results <-  c(mod.pred.results, 0)
  }
  mod.pred.results <- c(mod.pred.results, 2)
}

max(modified.preds[,2])

# Compare error rate here 
sum(mod.pred.results - test.data$result)
