import numpy
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.double


''' honeycomb lattice '''
class HoneycombLattice():
    def __init__(self, L):
        self.L = L # lattice (linear) size
        # site map
        self.site = {p:i for i, p in enumerate((x,y,a) 
            for x in range(L) for y in range(L) for a in range(2))}
        self.N = len(self.site) # number of sites
        # bond set
        self.bonds = set()
        for x in range(self.L):
            for y in range(self.L):
                i = self.site[(x,y,1)]
                j = self.site[(x,y,0)]
                self.bonds.add((i,j))
                self.bonds.add((j,i))
                j = self.site[((x+1)%self.L,y,0)]
                self.bonds.add((i,j))
                self.bonds.add((j,i))
                j = self.site[(x,(y+1)%self.L,0)]
                self.bonds.add((i,j))
                self.bonds.add((j,i))

    def __repr__(self):
        return type(self).__name__ + '({:d}x{:d}, {:d} sites, {:d} bonds)'.format(self.L, self.L, self.N, len(self.bonds))

    @property
    def adjacency_matrix(self):
        result = torch.zeros((self.N,self.N), dtype=dtype, device=device)
        for i,j in self.bonds:
            result[i,j] = 1.
        return result

    @property
    def stagger_matrix(self):
        result = torch.zeros((self.N,self.N), dtype=dtype, device=device)
        for x in range(self.L):
            for y in range(self.L):
                for a in range(2):
                    i = self.site[(x,y,a)]
                    result[i,i] = (-1)**a
        return result

''' meanfield model '''
class MFModel():
    def __init__(self, lattice):
        self.lattice = lattice
        self.I = torch.tensor([[1.,0.],[0.,1.]], dtype=dtype, device=device)
        self.X = torch.tensor([[0.,1.],[1.,0.]], dtype=dtype, device=device)
        self.H0 = - torch.kron(self.lattice.adjacency_matrix, self.I)
        self.H1 = torch.kron(self.lattice.stagger_matrix, self.X)

    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.lattice) + ')'

    def H(self, phi):
        return self.H0 + phi*self.H1

    def M(self, phi):
        H = self.H(phi)
        E, U = torch.linalg.eigh(H)
        M = U[:,:(H.shape[-1]//2)].contiguous()
        # add spin freedom
        M = torch.kron(M, self.I)
        return M

# sign and logdet of configurations xs on the MF state
def slogdet(M, x):
    M = M.unsqueeze(0).expand(x.shape[:1]+M.shape)
    D = M.gather(1, x.unsqueeze(-1).expand(x.shape + M.shape[-1:]))
    sign, logdet = torch.linalg.slogdet(D)
    W = M @ torch.linalg.inv(D)
    return sign, logdet, W

# update W matrix under x_b -> l
def W_update(W, l, b):
    W = W.clone()
    idx_b = b.view(-1,1,1).expand(W.shape[:-1]+(1,))
    Wb = W.gather(-1, idx_b)
    idx_l = l.view(-1,1,1)
    Wlb = Wb.gather(-2, idx_l)
    Wb = Wb/Wlb
    idx_l = l.view(-1,1,1).expand(W.shape[:-2]+(1,W.shape[-1]))
    Wl = W.gather(-2, idx_l)
    W = W - Wl * Wb
    W = W.scatter_add(-1, idx_b, Wb)
    return W

''' many-body basis '''
class MBBasis():
    def __init__(self, lattice):
        self.N = lattice.N
        self.Nf = 4
        self.Nmode = self.N * self.Nf
        # prepare charge list, hosts charge vectors for each mode
        qm = [(1,(-1)**l,(-1)**s) for i in range(self.N) for l in range(2) for s in range(2)]
        # establish a dictionary to collect mode indices for each distinct charge vector
        qdict = {}
        for m, q in enumerate(qm):
            if q in qdict:
                qdict[q].append(m)
            else:
                qdict[q] = [m]
        # determing how many modes to sample in each charge sector
        mode_count = numpy.stack([numpy.array(q) * len(ms) for q, ms in qdict.items()]).T
        expect_occ = numpy.array([self.Nmode//2, 0, 0]) 
        p = numpy.linalg.lstsq(mode_count, expect_occ, rcond=None)[0]
        self.Nq = (p/p.sum()*(self.Nmode//2)).astype(int)
        self.mlst = [numpy.array(ms) for ms in qdict.values()]
        self.qm = torch.tensor(qm, dtype=dtype, device=device)

    def sample(self, nsample=1):
        result = []
        for _ in range(nsample):
            x = []
            for n, ms in zip(self.Nq, self.mlst):
                x.append(numpy.random.choice(ms, n, replace=False))
            result.append(numpy.concatenate(x))
        return torch.tensor(numpy.stack(result))

    def q(self, x):
        qm = self.qm.unsqueeze(0).expand(x.shape[:1]+self.qm.shape)
        idx = x.unsqueeze(-1).expand(x.shape+self.qm.shape[-1:])
        qx = qm.gather(-2, idx)
        i = idx.div(self.Nf, rounding_mode='floor')
        q = torch.zeros([x.shape[0], self.N, qx.shape[-1]], dtype=dtype, device=device)
        q = q.scatter_add(-2, i, qx)
        return q

























