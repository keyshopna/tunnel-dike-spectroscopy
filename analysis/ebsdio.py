import numpy as np, pandas as pd, re
def read_ctf(path):
    with open(path,'rb') as f:
        head=[]; 
        for i,l in enumerate(f):
            l=l.decode('latin-1').rstrip('\r\n'); head.append(l)
            if l.startswith('Phase\tX\tY'): n=i; break
    h={}
    for l in head:
        p=l.split('\t')
        if len(p)>=2 and p[0] in ('XCells','YCells','XStep','YStep'): h[p[0]]=float(p[1])
    k=[i for i,l in enumerate(head) if l.startswith('Phases\t')][0]
    nph=int(head[k].split('\t')[1])
    phases={i+1:(head[k+1+i].split('\t')[2], int(head[k+1+i].split('\t')[3])) for i in range(nph)}
    df=pd.read_csv(path,sep='\t',skiprows=n,encoding='latin-1')
    nx,ny=int(h['XCells']),int(h['YCells'])
    df=df.iloc[:nx*ny]
    arr={c:df[c].to_numpy().reshape(ny,nx) for c in ['Phase','Euler1','Euler2','Euler3','MAD','BC','BS','Bands']}
    return h,phases,arr
