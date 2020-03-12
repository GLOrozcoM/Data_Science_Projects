# Module pour travailler sur les données des exoplanètes

kepler <- read.csv("kepler.csv")

kepler.clean <- na.omit(kepler)

dim(kepler)
dim(kepler.clean)

# Cela me dit que je ne peux pas juste enelver les rangs avec NA. 

sum(is.na(kepler$X..name))
table(kepler$X..name)

# Donne les variables et la somme de NA's pour chacun. 
somme.nas <- c()
for(variable.index in 1:length(names(kepler))){
  
  nom.character <- names(kepler)[variable.index]
  nombre.nas <- sum(is.na(kepler[variable.index]))
  
  somme.nas <- c(somme.nas, nom.character, nombre.nas)
}
somme.nas
# Pas tellement utile, trop de données. 

somme.nas <- c()
variables.guarder <- c()
for(variable.index in 1:length(names(kepler))){
  
  nombre.nas <- sum(is.na(kepler[variable.index]))
  
  if(nombre.nas <= 1/10 * nrow(kepler)){
    
    nom.character <- names(kepler)[variable.index]
    variables.guarder <- c(variables.guarder, nom.character)
    
    somme.nas <- c(somme.nas, nom.character, nombre.nas)
    
  }
  
  
}
somme.nas
variables.guarder

kepler.clean <- subset(kepler, select = variables.guarder)
names(kepler.clean)

kepler.clean <- na.omit(kepler.clean)

summary(kepler.clean)

function(){
  
}
