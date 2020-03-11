# Jouons avec Exo
kep <- read.csv("kepler.csv")
attach(kep)

# La table marche bien ?
sn.counts <- table(star_name)

# Prends les données pour les étoiles qui sont répétées
sn.rep <- sn.counts[sn.counts > 1]

# En utilisant les noms...
k.sys <- kep[-c(1:3732),]
for(i in 1:length(star_name)){
  if( star_name[i] %in% names(sn.rep) ){
   k.sys <- rbind(kep[i,],k.sys) 
  }
}

# Est-ce-que ce nouveau DF me donne des planètes qui sont des sytèmes ?

# -- Comment vérifier ? Tout, mon amie, tout. 
# --- Il doit y avoir plusieurs étoiles avec le même nom. Il n'y a pas d'étoile 
# --- avec une seule entrée. 
new.counts <- table(k.sys$star_name)
sum(new.counts == 1)
sum(new.counts == 0)
sum(new.counts == 2)

# -- Aussi la même chose pour le nombre des étoiles qui est répétée. 
length(unique(k.sys$star_name))

# Voyons voir les dimensions
dim(k.sys)

# Là on peut voir le nombre des exo- planètes dans chaque système.
new.counts[new.counts > 1]

# On voudras savoir s'il y en existe des sytèmes avec 9 planètes comme le notre. 
new.counts[new.counts == 9]

# -- Dommage, mais on dit par fois que pluto n'est pas une planète donc...
new.counts[new.counts == 8]
# Kepler-90 
# 8 

# Et quels sont les noms de planètes de ce système ?
kepler.90 <- subset(k.sys, star_name == "Kepler-90")
dim(kepler.90)
kepler.90$X..name
# Il y avait déjà des noms de l'étoile la dedans...

# Quel est l'âge de l'étoile ?
kepler.90$star_age

# -- Eh bon, combien de nas?
sum(is.na(kepler.90$star_age))
# On n'a pas l'âge...

# Les noms de nos planètes 
k.sys$X..name

# Beaucoup des noms contiennent Kepler. Peut-il que la convention d'une planète 
# détermine le nom ? ...

# Est-ce-qu'on peut savoir la distance entre le systemè et la terre ?
# -- comment chercher pour un caractère avec lettres
names(k.sys)

# Nous avons le caractère pour la distance, mais on sait pais si la distance entre 
# la terre et l'étoile ou si c'est la distance entre l'étoile et la planète.
min(k.sys$star_distance, na.rm = T)
# et savoir aussi si c'est en parsecs...je voudrais savoir ce que ça signifie...

