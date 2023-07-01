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

    def sample(self):
        x = []
        for n, ms in zip(self.Nq, self.mlst):
            x.append(numpy.random.choice(ms, n, replace=False))
        x = torch.tensor(numpy.concatenate(x))
        return x

    def q(self, x):
        qx = self.qm[x]
        i = x.div(self.Nf, rounding_mode='floor')
        i = i.unsqueeze(-1).expand([i.shape[0], qx.shape[-1]])
        q = torch.zeros([self.N, qx.shape[-1]], dtype=dtype, device=device)
        q[:,0] = - self.Nf//2 # bias to total charge
        q = q.scatter_add(0, i, qx)
        return q

''' mean field state '''
class MFState():
    def __init__(self, M, x):
        self.M = M
        self.x = x
        self._sign = None
        self._logdet = None
        self._W = None
        self._motion = None

    def __repr__(self):
        return type(self).__name__ + '({})'.format(self.x.tolist())

    def clone(self):
        new = type(self)(self.M, self.x.clone())
        if self._sign is not None:
            new._sign = self._sign.clone()
        if self._logdet is not None:
            new._logdet = self._logdet.clone()
        if self._W is not None:
            new._W = self._W.clone()
        return new 

    @property
    def sign(self):
        if self._sign is None:
            self._sign, self._logdet = torch.linalg.slogdet(self.M[self.x])
        return self._sign

    @property
    def logdet(self):
        if self._logdet is None:
            self._sign, self._logdet = torch.linalg.slogdet(self.M[self.x])
        return self._logdet

    @property
    def det(self):
        return self.sign * self.logdet.exp()

    @property
    def W(self):
        if self._W is None:
            self._W = self.M @ torch.linalg.inv(self.M[self.x])
        return self._W

    @property
    def motion(self):
        if self._motion is None:
            weight = self.W**2
            mode = torch.multinomial(weight.sum(0),1)[0]
            target = torch.multinomial(weight[:,mode],1)[0]
            self._motion = (mode, target)
        return self._motion

    def update(self, mode, target):
        if self._motion is None:
            self._motion = (mode, target)
        self.x[mode] = target
        ratio = self.W[target,mode]
        self._sign = self.sign * ratio.sign()
        self._logdet = self.logdet + ratio.abs().log()
        return self

    def accept(self):
        mode, target = self.motion
        ratio = self.W[target,mode]
        Wcol = self.W[:,mode] / ratio
        Wrow = self.W[target,:]
        self._W -= Wrow.unsqueeze(0) * Wcol.unsqueeze(1)
        self._W[:,mode] += Wcol
        self._motion = None
        return self

    def propose(self, steps=1):
        new = self.clone()
        for step in range(steps):
            if new._motion is not None:
                new.accept()
            mode, target = new.motion
            new.update(mode, target)
        return new

    def forward(self, motions):
        new = self.clone()
        for mode, target in motions:
            if new._motion is not None:
                new.accept()
            new.update(mode, target)
        return new

    




















