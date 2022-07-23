import os
import centroid_w2v


# Appel a la classe
c = centroid_w2v.CentroidSummarizer()

for file in os.listdir('C://Users//HP//Desktop//Résumeur_automatique//Articles//topics'):
    for element in os.listdir('C://Users//HP//Desktop//Résumeur_automatique//Articles//topics//'+file):
        f=open('C://Users//HP//Desktop//Résumeur_automatique//Articles//topics//'+file+'//'+element,'r').read()
        # Fixation de la taille limite du résumé a 30%
        limit = ((30 * len(f.split('.'))) / 100)
        #Résumé
        res=c.summarize(text=f,limit=limit)
        f=open('C://Users//HP//Desktop//Résumeur_automatique//mesres_centroid_synop//'+file+'.txt','w',encoding='utf8')
        f.write(res)
        f.close()