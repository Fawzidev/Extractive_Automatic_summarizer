import os
import centroid_w2v

c = centroid_w2v.CentroidSummarizer()

for file in os.listdir('topics'):
    for element in os.listdir('topics//'+file):
        f=open('topics//'+file+'//'+element,'r').read()
        
        limit = ((30 * len(f.split('.'))) / 100)
        
        res=c.summarize(text=f,limit=limit)
        f=open('//mesres_centroid_synop//'+file+'.txt','w',encoding='utf8')
        f.write(res)
        f.close()