import random
import copy
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('mps')
real_type = torch.double
int_type = torch.long

''' autograd functions ------- '''
''' eigenvector solver with stable autograd '''
class EigVecsH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, UPLO, tol):
        L, Q = torch.linalg.eigh(A, UPLO=UPLO)
        ctx.save_for_backward(Q)
        ctx.L = L
        ctx.requires_grad_A = A.requires_grad
        ctx.tol = tol
        return Q
    
    @staticmethod
    def backward(ctx, grad_Q):
        (Q,) = ctx.saved_tensors
        if ctx.requires_grad_A:
            QH = Q.T.conj()
            G = QH.mm(grad_Q)
            G = (G - G.T.conj())/2
            dL = ctx.L.unsqueeze(0) - ctx.L.unsqueeze(1)
            F = dL / (dL**2 + ctx.tol**2)
            grad_A = Q.mm(G * F).mm(QH)
            return grad_A, None, None
        else:
            return None, None, None

def eigvecsh(A, UPLO='L', tol=1.e-10):
    # tol sets the resolution of eigenvalues
    return EigVecsH.apply(A, UPLO, tol)

''' update W matrix with autograd '''
class UpdateW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, mode_tgt, idx):
        ratio = W[mode_tgt, idx].clone()
        Wcol = W[:, idx].clone() # idx -> beta
        Wrow = W[mode_tgt, :].clone() # tgt -> l
        Wrow[idx] -= 1.
        ctx.need_grad = W.requires_grad
        if ctx.need_grad:
            ctx.mode_tgt, ctx.idx, ctx.ratio, ctx.Wcol, ctx.Wrow = mode_tgt, idx, ratio, Wcol, Wrow
        return W - Wrow.unsqueeze(0) * Wcol.unsqueeze(1) / ratio
        
    @staticmethod
    def backward(ctx, grad_W):
        grad_Wout = None
        if ctx.need_grad:
            grad_Wcol = grad_W @ ctx.Wrow / ctx.ratio
            grad_Wrow = ctx.Wcol @ grad_W / ctx.ratio
            grad_ratio = grad_Wrow @ ctx.Wrow / ctx.ratio
            grad_Wout = grad_W.clone()
            grad_Wout[:, ctx.idx] -= grad_Wcol
            grad_Wout[ctx.mode_tgt, :] -= grad_Wrow
            grad_Wout[ctx.mode_tgt, ctx.idx] += grad_ratio
        return grad_Wout, None, None
    
updateW = UpdateW.apply

''' operator '''
class Operator():
    def __init__(self, terms=None):
        self.terms = dict() if terms is None else terms

    def __repr__(self):
        return type(self).__name__ + '({} terms)'.format(len(self))

    def __len__(self):
        return len(self.terms)

    def __neg__(self):
        return self * (-1.)

    def __add__(self, other):
        if isinstance(other, Operator):
            if len(self) >= len(other):
                for key, val in other.terms.items():
                    if key in self.terms:
                        self.terms[key] += val
                    else:
                        self.terms[key] = val
                    if abs(self.terms[key]) < 1.e-12:
                        del self.terms[key]
                return self
            else:
                return other + self
        else:
            print(other, isinstance(other, Operator))
            raise NotImplementedError

    def __rmul__(self, other):
        for key in self.terms.keys():
            self.terms[key] *= other
        return self

    def __mul__(self, other):
        for key in self.terms.keys():
            self.terms[key] *= other
        return self

    '''def evaluate_sqr(self, state):
        result = 0.
        for key1, val1 in self.terms.items():
            try:
                state1 = state.forward(key1)
                for key2, val2 in self.terms.items():
                    try:    
                        state2 = state1.forward(key2)
                        ratio = state2.det / state.det
                        ratio *= torch.exp((state.energy - state2.energy)/2)
                        result += ratio * val1 * val2
                    except ConfigurationError:
                        pass
            except ConfigurationError:
                pass
        return result'''
    
    def evaluate(self, state):
        result = torch.tensor(0., dtype=real_type, device=device)
        for key, val in self.terms.items():
            try:
                state1 = state.forward(key) # if key involes more than 1 hopping step, could raise state1.det -> nan
                det = state1.det
                if torch.isnan(det):
                    state1._sign, state1._logdet = None, None
                    det = state1.det
                ratio = det / state.det
                ratio *= torch.exp((state.energy - state1.energy)/2)
                result += ratio * val
                #else:
                    #print(key, state1.logdet)
            except ConfigurationError:
                pass
        return result

''' lattice model '''
class LatticeModel():
    def __init__(self, L, Nf=4):
        self.L = L # lattice size
        self.Nf = Nf # flavor number
        self.siteidx = {site:idx for idx, site in enumerate(self.sites())}
        self.Ns = len(self.siteidx) # site number
        self.N = self.Ns *  self.Nf # total number of modes
        
    def sites(self):
        raise NotImplementedError
                    
    def bonds(self):
        raise NotImplementedError
                    
    def mode(self, site, flavor):
        if isinstance(site, tuple):
            site = self.siteidx[site]
        if isinstance(flavor, tuple):
            flavor = sum(f*self.Nf//2**(k+1) for k,f in enumerate(flavor))
        return self.Nf * site + flavor
    
    def hopping(self, site0, site1):
        op = Operator()
        for flavor in range(self.Nf):
            mode0 = self.mode(site0, flavor)
            mode1 = self.mode(site1, flavor)
            op = op + Operator({((mode0, mode1),): 1., ((mode1, mode0),): 1.})
        return op
        
    def spin_exchange(self, site):
        op = Operator()
        for l in range(2):
            uppath = (self.mode(site, (l,0)), self.mode(site, (1-l,0)))
            dnpath = (self.mode(site, (1-l,1)), self.mode(site, (l,1)))
            op = op + Operator({(uppath, dnpath): -0.5})
        return op
    
    def spin_coupling(self, site):
        op = Operator()
        for s0 in range(2):
            mode0 = self.mode(site, (0,s0))
            for s1 in range(2):
                mode1 = self.mode(site, (1,s1))
                sign = (-1)**(s0+s1)
                op = op + Operator({((mode0,mode0), (mode1,mode1)): 0.25*sign})
        return op
    
    def inter_layer(self, site):
        op = Operator()
        for s in range(2):
            mode0 = self.mode(site, (0,s))
            mode1 = self.mode(site, (1,s))
            op = op + Operator({((mode0, mode1),): 1., ((mode1, mode0),): 1.})
        return op
    
    @property
    def Ht(self):
        op = Operator()
        for site1, site2, val in self.bonds():
            op = op + val * self.hopping(site1, site2)
        return op
    
    @property
    def HJ(self):
        op = Operator()
        for site in self.sites():
            op = op + self.spin_exchange(site) + self.spin_coupling(site)
        return op
    
    @property
    def Hu(self):
        op = Operator()
        for site in self.sites():
            (x, y, a) = site
            op = op + (-1)**a * self.inter_layer(site)
        return op

    # matrix representation of fermion bilinear operator
    def bilinear_matrix(self, op):
        out = torch.zeros((self.N, self.N), dtype=real_type, device=device)
        for key in op.terms.keys():
            if len(key) == 1:
                i, j = key[0]
                out[j,i] = op.terms[key]
            else:
                raise RuntimeError('bilinear_matrix encounters non-bilinear operator.')
        return out

''' honeycomb lattice model '''
class HoneycombModel(LatticeModel):
    def sites(self):
        for x in range(self.L):
            for y in range(self.L):
                for a in range(2):
                    yield((x, y, a))
                    
    def bonds(self):
        for (x, y, a) in self.sites():
            if a == 0:
                yield(((x, y, 0), (x, y, 1), 1.))
                if self.L > 1:
                    yield(((x, y, 0), ((x+1)%self.L, y, 1), 1.))
                    yield(((x, y, 0), (x, (y+1)%self.L, 1), 1.))

''' occupation configuration '''
class Configuration(list):
    def __init__(self, config=[], modeidx=None):
        super(type(self), self).__init__(config)
        if modeidx is None:
            self.modeidx = {mode:idx for idx,mode in enumerate(self)}
            self.validate()
        else:
            self.modeidx = modeidx

    def validate(self, mode=None):
        if len(self) != len(self.modeidx):
            action = None if mode is None else 'created'
            raise ConfigurationError(action=action, mode=mode)
    
    def __contains__(self, mode):
        return mode in self.modeidx # use dict for O(1) lookup
    
    def __setitem__(self, idx, mode):
        self.modeidx[mode] = self.modeidx.pop(self[idx])
        super(type(self), self).__setitem__(idx, mode)
        self.validate(mode)
    
    def clone(self):
        return type(self)(self, copy.copy(self.modeidx))
    
    def replace(self, mode_src, mode_tgt):
        try:
            idx = self.modeidx.pop(mode_src)
        except KeyError:
            raise ConfigurationError(action='annihilated', mode=mode_src) from None
        self.modeidx[mode_tgt] = idx
        super(type(self), self).__setitem__(idx, mode_tgt)
        self.validate(mode_tgt)
        return self

    def append(self, mode):
        self.modeidx[mode] = len(self)
        super(type(self), self).append(mode)
        self.validate(mode)
        return self

class ConfigurationError(Exception):
    def __init__(self, action=None, mode=None):
        if action is None:
            message = 'Mode collision occured.'
        else:
            message = 'Mode {} can not be {} in the configuration.'.format(mode, action)
        super(type(self), self).__init__(message)

''' basis system '''
class BasisSystem():
    def __init__(self, lattice, Nf=4):
        self.lattice = lattice
        self.Ns = lattice.Ns # site number
        self.Nf = Nf # flavor number
        # prepare charge list, hosting charge vectors for each mode
        qs = [((-1)**l,(-1)**s) for i in range(self.Ns) for l in range(2) for s in range(2)]
        self.qdict = {}
        for m, q in enumerate(qs):
            if q in self.qdict:
                self.qdict[q].append(m)
            else:
                self.qdict[q] = [m]
        self.qs = torch.tensor(qs, dtype=int_type, device=device)
        self.dq = self.qs.unsqueeze(0) - self.qs.unsqueeze(1)
    
    def sample(self):
        config = Configuration()
        for modes in self.qdict.values():
            for mode in random.sample(modes, len(modes)//2):
                config.append(mode)
        return config
    
    def qi(self, config):
        qx = self.qs[config]
        i = torch.tensor(config, dtype=int_type, device=device).div(self.Nf, rounding_mode='floor')
        i = i.unsqueeze(-1).expand([i.shape[0], qx.shape[-1]])
        qi = torch.zeros([self.Ns, qx.shape[-1]], dtype=int_type, device=device)
        qi = qi.scatter_add(0, i, qx)
        return qi

    def qmsk(self, config, q0=None):
        dq = self.dq[:,config,:]
        if q0 is not None:
            dq = dq - q0.view(1,1,-1)
        qmsk = (dq==0).all(-1).to(dtype=real_type)
        return qmsk

''' mean-field model '''
class MeanfieldModel(torch.nn.Module):
    def __init__(self, lattice):
        super().__init__()
        self.H0 = - lattice.bilinear_matrix(lattice.Ht)
        self.H1 = lattice.bilinear_matrix(lattice.Hu)
        self.u = torch.nn.Parameter(torch.tensor(1., dtype=real_type, device=device))
        self.reset()

    def reset(self):
        self._M = None
        return self

    @property
    def H(self):
        return self.H0 + self.u * self.H1

    @property
    def M(self):
        if self._M is None:
            H = self.H
            U = eigvecsh(H)
            self._M = U[:,:(H.shape[-1]//2)].contiguous()
        return self._M

    def slogdet(self, config):
        return torch.linalg.slogdet(self.M[config])

    def W(self, config):
        return self.M @ torch.linalg.inv(self.M[config])

''' projector model '''
class ProjectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2,3), dtype=real_type, device=device))

    def energy(self, qi):
        mu = - torch.nn.functional.log_softmax(self.weight, -1)
        idx = qi.abs()
        return sum(mu[i,idx[:,i]].sum() for i in range(2))
        # return 0.

''' variational model '''
class VariationalModel(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        self.lattice = HoneycombModel(L)
        self.basis = BasisSystem(self.lattice)
        self.meanfield = MeanfieldModel(self.lattice)
        self.projector = ProjectorModel()
        self.reset()

    def reset(self):
        self._step = 0
        self._rejects = 0
        self.meanfield.reset()
        return self

    def state(self, config=None):
        if config is None:
            config = self.basis.sample()
        return VariationalState(self, Configuration(config))

    def propose1(self, state):
        trans_prob = state.trans_prob(0) # q0 = 0 on state0
        valid = False
        while not valid:
            try:
                idx = torch.multinomial(trans_prob.sum(0), 1).item()
                mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
                mode_src = state.config[idx]
                if mode_tgt == mode_src:
                    return state, torch.tensor(0., dtype=real_type, device=device)
                else:
                    logq = trans_prob[mode_tgt, idx].log()
                    state = state.clone() # state0 -> state1
                    state.replace(mode_src, mode_tgt)
                    trans_prob = state.trans_prob(0) # q0 = 0 on state1
                    logq = logq - trans_prob[mode_src, idx].log()
                    valid = True
            except ConfigurationError:
                pass
            return state, logq

    def propose2(self, state):
        logq = torch.tensor(0., dtype=real_type, device=device)
        trans_prob = state.trans_prob() # q0 = any on state0
        valid = False
        while not valid:
            try:
                idx = torch.multinomial(trans_prob.sum(0), 1).item()
                mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
                mode_src = state.config[idx]
                dq = self.basis.dq[mode_tgt, mode_src] # compute dq
                logq = logq + trans_prob[mode_tgt, idx].log()
                if mode_tgt != mode_src:
                    state = state.clone() # state0 -> state1
                    state.replace(mode_src, mode_tgt)
                    valid = True
            except ConfigurationError:
                pass
        trans_prob = state.trans_prob(-dq) # q0 = -dq on state1
        logq = logq - trans_prob[mode_src, idx].log()
        valid = False
        while not valid:
            try:
                idx = torch.multinomial(trans_prob.sum(0), 1).item()
                mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
                mode_src = state.config[idx]
                logq = logq + trans_prob[mode_tgt, idx].log()
                if mode_tgt != mode_src:
                    state = state.clone() # state1 -> state2
                    state.replace(mode_src, mode_tgt)
                    valid = True
            except ConfigurationError:
                pass
        trans_prob = state.trans_prob() # q0 = any on state2
        logq = logq - trans_prob[mode_src, idx].log()
        return state, logq
        
    '''def MCstep_pg(self, state):
        self._step += 1
        if self._step % 5 == 0:
            propose = self.propose2
        else:
            propose = self.propose1
        new_state, logq = propose(state)
        new_logprob = new_state.logprob
        old_logprob = state.logprob
        logq = new_logprob.detach() - old_logprob.detach() #- logq
        if torch.exp(logq) > random.random():
            new_prob = new_logprob.exp()
            old_prob = old_logprob.exp()
            return new_state, new_prob / (new_prob + old_prob)
        else:
            self._rejects += 1
            return state, 0. # return -plcy is working but worse

    def MCrun_pg(self, H=None, state=None, steps=1):
        if state is None: # no state provided, sample a random one
            state = self.state()
        rwd = torch.tensor(0., dtype=real_type, device=device)
        Hval = torch.tensor(0., dtype=real_type, device=device)
        for _ in range(steps):
            state, plcy = self.MCstep_pg(state)
            if H is not None:
                Heva = H.evaluate(state)
                rwd += Heva.detach() * plcy
                Hval += Heva
        if H is None:
            return state
        else:
            return rwd/steps, Hval/steps, state
            
    def exct_run(self, H=None, configs=None):
        Hval = torch.tensor(0., dtype=real_type, device=device)
        norm = torch.tensor(0., dtype=real_type, device=device)
        for config in configs:
            norm += self.state(config).logprob.exp()
        for config in configs:
            Hval += H.evaluate(self.state(config)) * self.state(config).logprob.exp() / norm
        return Hval ,configs''' 
        
    def MCstep_sr(self, state):
        self._step += 1
        if self._step % 5 == 0:
            propose = self.propose2
        else:
            propose = self.propose1
        new_state, logq = propose(state)
        new_logprob = new_state.logprob
        old_logprob = state.logprob
        logq = new_logprob.detach() - old_logprob.detach() #-logq
        if torch.exp(logq) > random.random():
            return new_state, new_logprob
        else:
            self._rejects += 1
            return state, old_logprob

    def MCrun_sr(self, H=None, state=None, steps=1):
        if state is None: # no state provided, sample a random one
            state = self.state()
        Hval = torch.tensor(0., dtype=real_type, device=device)
        ls = []
        logprobs = []
        for _ in range(steps):
            state, logprob = self.MCstep_sr(state)
            if H is not None:
                Heva = H.evaluate(state).detach()
                Hval += Heva
                ls.append(Heva)
                logprobs.append(logprob)
        if H is None:
            return state
        else:
            Hval = Hval/steps
            obj = torch.tensor(0., dtype=real_type, device=device)
            for l, logprob in zip(ls, logprobs):
                obj += (l - Hval) * logprob
            return  obj/steps, Hval, state
        
''' variational state '''
class VariationalState():
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.reset()

    def __repr__(self):
        return type(self).__name__ + '({})'.format(self.config)

    def reset(self):
        self._sign = None
        self._logdet = None
        self._W = None
        self._temp = None 
        self._energy = None
        return self

    def clone(self):
        new = type(self)(self.model, self.config.clone())
        # clone/link any cache variables that is not None
        if self._sign is not None:
            new._sign = self._sign.clone()
        if self._logdet is not None:
            new._logdet = self._logdet.clone()
        if self._W is not None:
            new._W = self._W.clone()
        if self._temp is not None:
            new._temp = self._temp
        if self._energy is not None:
            new._energy = self._energy # no need to clone
        return new 

    @property
    def sign(self):
        if self._sign is None:
            self._sign, self._logdet = self.model.meanfield.slogdet(self.config)
        return self._sign

    @property
    def logdet(self):
        if self._logdet is None:
            self._sign, self._logdet = self.model.meanfield.slogdet(self.config)
        return self._logdet

    @property
    def det(self):
        return self.sign * self.logdet.exp()

    @property
    def qi(self):
        return self.model.basis.qi(self.config)

    @property
    def energy(self):
        if self._energy is None:
            self._energy = self.model.projector.energy(self.qi)
        return self._energy

    @property
    def logprob(self):
        return 2*self.logdet - self.energy

    @property
    def W(self):
        if self._W is None: # W has not been initialized
            self._W = self.model.meanfield.W(self.config)
        if self._temp is not None: # W has not been updated
            self._W = updateW(self._W, *self._temp) # update W
            self._temp = None # clear temp
        return self._W
    
    def replace(self, mode_src, mode_tgt):
        try:
            idx = self.config.modeidx[mode_src]
        except KeyError:
            raise ConfigurationError(action='annihilated', mode=mode_src) from None
        if mode_src != mode_tgt:
            ratio = self.W[mode_tgt, idx]
            self._sign = self.sign * ratio.sign()
            self._logdet = self.logdet + ratio.abs().log() 
            # config should NOT be modified before evaluation of slogdet
            self.config[idx] = mode_tgt
            self._temp = (mode_tgt, idx) # cache temp data
            self._energy = None # energy should be recalculated
        return self

    def forward(self, instruction):
        new = self.clone()
        for mode_src, mode_tgt in instruction:
            try:
                new.replace(mode_src, mode_tgt)
            except ConfigurationError as err:
                raise err from None
        return new

    def backward(self, instruction):
        new = self.clone()
        for mode_src, mode_tgt in reversed(instruction):
            try:
                new.replace(mode_tgt, mode_src)
            except ConfigurationError as err:
                raise err from None
        return new

    def trans_prob(self, q0=None):
        weight = self.W.detach()**2
        if q0 is not None:
            if q0 is 0:
                weight = weight * self.model.basis.qmsk(self.config)
            else:
                weight = weight * self.model.basis.qmsk(self.config, q0)
        trans_prob = weight / weight.sum()
        d = trans_prob.max().log10().round().abs() + 3
        trans_prob = trans_prob.round(decimals = int(d)).ceil()
        return trans_prob

def test(H,mdl):
    configs = [[0, 1, 2, 3],
     [0, 1, 2, 7]
     ,[0, 1, 3, 6]
     ,[0, 1, 6, 7]
     ,[0, 2, 3, 5]
     ,[0, 2, 5, 7]
     ,[0, 3, 4, 7]
     ,[0, 3, 5, 6]
     ,[0, 5, 6, 7]
     ,[1, 2, 3, 4]
     ,[1, 2, 4, 7]
     ,[1, 2, 5, 6]
     ,[1, 3, 4, 6]
     ,[1, 4, 6, 7]
     ,[2, 3, 4, 5]
     ,[2, 4, 5, 7]
     ,[3, 4, 5, 6]
     ,[4, 5, 6, 7]
    ]
    ruler = [-0.0399179,
     -0.0780561
     ,0.0780561
     ,-0.0399179
     ,0.0780561 
     ,0.0302494
     ,0.470195
     ,-0.500444
     ,-0.0780561
     ,0.0780561
     ,-0.500444
     ,0.470195
     ,0.0302494
     ,0.0780561
     ,-0.0399179
     ,-0.0780561
     ,0.0780561
     ,-0.0399179
    ]
    ampls = []
    Hval = 0
    for config in configs:
        ampls.append(- (torch.exp(- mdl.state(config).energy / 2) * mdl.state(config).det).item())
    norm = sum(ampl**2 for ampl in ampls)
    ampls = [ampl / norm**0.5 for ampl in ampls]
    for config, ampl in zip(configs, ampls):
        print('{} {}'.format(config, ampl))
        Hval += H.evaluate(mdl.state(config)) * ampl**2
    fid =  sum([x*y for x,y in zip(ruler,ampls)])
    print('energy =', Hval.item(), 'normalization factor =', norm, 'fidelity =', abs(fid))
    
def nantest(t):
    if torch.any(torch.isnan(t)):
        raise RuntimeError('A-oh, nan detected.')
def zerotest(t):
    if torch.any(t) == 0:
        raise RuntimeError('A-oh, zero detected.')