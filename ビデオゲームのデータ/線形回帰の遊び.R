# 線形回帰のモデルで遊ぶためのファイル

games <- read.csv( "Video_Game_Sales_Clean.csv" )

names(games)

# Y = Global_Sales, x1 = Critic_Score, x2 = User_Score
lm.fit = lm(Global_Sales ~ Critic_Score + User_Score, data = games)
lm.fit

# Coefficients:
#   (Intercept)  Critic_Score    User_Score  
# -1.28537       0.03976      -0.10179  

# 母数が見えた（上）。

summary(lm.fit)

# p-value（2.2e-16）を見たら、これがいいモデルではないと思う。
# R^2（１と近くの値が欲しい）は0.05985ので、いいフィットではない。

# Y = Global_Sales, x1 = Critic_Score
lm.fit = lm(Global_Sales ~ Critic_Score, data = games)
lm.fit

# Coefficients:
#   (Intercept)  Critic_Score  
# -1.58580       0.03363  

summary(lm.fit)

# ここでもいいモデルだとは言えないらしい。

## Rのデータセットで遊ぶ
names(trees)

lm.fit <- lm(Volume ~ Girth + Height, data = trees)
lm.fit

# Coefficients:
#   (Intercept)        Girth       Height  
# -57.9877       4.7082       0.3393  

summary(lm.fit)

# Residual standard error: 3.882 on 28 degrees of freedom
# Multiple R-squared:  0.948,	Adjusted R-squared:  0.9442 
# F-statistic:   255 on 2 and 28 DF,  p-value: < 2.2e-16

# なので、いいモデールだよね。影響のことをみれば何がわかる？

