import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy import ndimage

GROUP = {1:'feldspar',4:'feldspar',5:'augite',7:'Fe-Ti oxide',8:'Fe-Ti oxide',3:'ilmenite',6:'apatite',2:'quartz'}
GROUP_ID = {'feldspar':1,'augite':2,'Fe-Ti oxide':3,'ilmenite':4,'apatite':5,'quartz':6}

def rot(axis, deg):
    t=np.radians(deg); c,s=np.cos(t),np.sin(t)
    if axis=='x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis=='y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def cubic_ops():
    import itertools
    ops=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1,-1],repeat=3):
            M=np.zeros((3,3))
            for i,p in enumerate(perm): M[i,p]=signs[i]
            if np.isclose(np.linalg.det(M),1): ops.append(M)
    return ops

def sym_ops(group, mono_axis='y'):
    I=np.eye(3)
    if group=='feldspar': return [I]
    if group=='augite': return [I, rot(mono_axis,180)]
    if group=='Fe-Ti oxide': return cubic_ops()
    if group in ('ilmenite','quartz'): return [rot('z',a) for a in (0,120,240)]
    if group=='apatite': return [rot('z',a) for a in range(0,360,60)]
    return [I]

def euler_to_g(e1,e2,e3):
    p1,P,p2=[np.radians(x) for x in (e1,e2,e3)]
    c1,s1,c2,s2,C,S=np.cos(p1),np.sin(p1),np.cos(p2),np.sin(p2),np.cos(P),np.sin(P)
    g=np.empty(p1.shape+(3,3))
    g[...,0,0]=c1*c2-s1*s2*C; g[...,0,1]=s1*c2+c1*s2*C; g[...,0,2]=s2*S
    g[...,1,0]=-c1*s2-s1*c2*C; g[...,1,1]=-s1*s2+c1*c2*C; g[...,1,2]=c2*S
    g[...,2,0]=s1*S; g[...,2,1]=-c1*S; g[...,2,2]=C
    return g

def fill_zero_solutions(arr, min_neighbors=5, iterations=2):
    """Fill unindexed pixels surrounded by >= min_neighbors indexed neighbours (8-connected),
    copying the phase + orientation of the most common neighbouring phase. Like AZtec/Channel5 noise reduction."""
    P=arr['Phase'].copy(); E=[arr['Euler1'].copy(),arr['Euler2'].copy(),arr['Euler3'].copy()]
    ny,nx=P.shape
    offs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(iterations):
        pad=np.pad(P,1)
        neigh=np.stack([pad[1+dy:1+dy+ny,1+dx:1+dx+nx] for dy,dx in offs])
        nidx=(neigh>0).sum(0)
        cand=(P==0)&(nidx>=min_neighbors)
        if not cand.any(): break
        # most common neighbour phase
        counts=np.stack([(neigh==k).sum(0) for k in range(1,9)])
        best=counts.argmax(0)+1
        ys,xs=np.nonzero(cand)
        newP=best[ys,xs]
        # copy orientation from first neighbour with that phase
        Epad=[np.pad(e,1) for e in E]
        newE=[np.zeros(len(ys)) for _ in range(3)]
        done=np.zeros(len(ys),bool)
        for dy,dx in offs:
            nb=pad[1+ys+dy, 1+xs+dx]
            take=(~done)&(nb==newP)
            for j in range(3): newE[j][take]=Epad[j][1+ys[take]+dy,1+xs[take]+dx]
            done|=take
        P[ys,xs]=newP
        for j in range(3): E[j][ys,xs]=newE[j]
    out=dict(arr); out['Phase']=P; out['Euler1'],out['Euler2'],out['Euler3']=E
    return out

def misorientation(gA,gB,ops):
    M=np.einsum('nij,nkj->nik',gB,gA)   # gB gA^T (crystal frame)
    best=np.full(len(M),-1.0)
    for S in ops:
        tr=np.einsum('ij,nji->n',S,M)
        best=np.maximum(best,tr)
    return np.degrees(np.arccos(np.clip((best-1)/2,-1,1)))

def reconstruct(arr, threshold=10.0, twin_merge=True, mono_axis='y'):
    P=arr['Phase']; ny,nx=P.shape
    G=np.zeros_like(P); 
    for k,g in GROUP.items(): G[P==k]=GROUP_ID[g]
    g=euler_to_g(arr['Euler1'],arr['Euler2'],arr['Euler3']).reshape(-1,3,3)
    idx=np.arange(ny*nx).reshape(ny,nx)
    rows,cols=[],[]
    for a,b in [(idx[:,:-1],idx[:,1:]),(idx[:-1,:],idx[1:,:])]:
        a=a.ravel(); b=b.ravel()
        Ga=G.ravel()[a]; Gb=G.ravel()[b]
        same=(Ga==Gb)&(Ga>0)
        a,b,Ga=a[same],b[same],Ga[same]
        keep=np.zeros(len(a),bool)
        for name,gid in GROUP_ID.items():
            m=Ga==gid
            if not m.any(): continue
            ang=misorientation(g[a[m]],g[b[m]],sym_ops(name,mono_axis))
            ok=ang<threshold
            if twin_merge and name=='feldspar': ok|=ang>180-threshold
            keep[np.nonzero(m)[0][ok]]=True
        rows.append(a[keep]); cols.append(b[keep])
    r=np.concatenate(rows); c=np.concatenate(cols); N=ny*nx
    A=coo_matrix((np.ones(len(r)),(r,c)),shape=(N,N))
    _,lab=connected_components(A,directed=False)
    lab=lab.reshape(ny,nx); lab[G==0]=-1
    return lab, G

def grain_table(lab, G, step, min_pixels=10):
    import pandas as pd
    valid=lab>=0
    ids,inv,counts=np.unique(lab[valid],return_inverse=True,return_counts=True)
    gid=np.zeros(len(ids),int); gid[inv]=G[valid]
    ys,xs=np.nonzero(valid)
    # second moments for long/short axes
    n=counts.astype(float)
    mx=np.bincount(inv,xs)/n; my=np.bincount(inv,ys)/n
    sxx=np.bincount(inv,xs*xs)/n-mx**2; syy=np.bincount(inv,ys*ys)/n-my**2; sxy=np.bincount(inv,xs*ys)/n-mx*my
    tr=sxx+syy; det=sxx*syy-sxy**2; disc=np.sqrt(np.maximum(tr**2/4-det,0))
    l1=tr/2+disc; l2=np.maximum(tr/2-disc,0)
    # touches map edge?
    ny,nx=lab.shape
    edge=np.zeros(len(ids),bool)
    border=np.zeros_like(valid); border[0,:]=border[-1,:]=border[:,0]=border[:,-1]=True
    eb=np.unique(inv[border[valid]]); edge[eb]=True
    inv_name={v:k for k,v in GROUP_ID.items()}
    df=pd.DataFrame({'mineral':[inv_name[x] for x in gid],'pixels':counts,
        'area_um2':counts*step**2,'ecd_um':2*np.sqrt(counts*step**2/np.pi),
        'long_axis_um':4*np.sqrt(l1+1/12)*step,'short_axis_um':4*np.sqrt(l2+1/12)*step,'touches_edge':edge,
        'x_px':mx,'y_px':my})
    return df[df.pixels>=min_pixels].reset_index(drop=True)
