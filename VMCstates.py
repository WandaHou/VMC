import random
import copy
import torch
import numpy
from matplotlib import pyplot
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

''' update Ws matrix with autograd '''
class UpdateWs(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Ws, need_updt, mode_tgts, idxs):
        l = len(need_updt)
        t_mode_tgts = torch.tensor(mode_tgts, dtype=int_type, device=device)
        t_idxs = torch.tensor(idxs, dtype=int_type, device=device)
        index = t_mode_tgts * Ws.shape[2] + t_idxs
        ratios = torch.gather(Ws[need_updt].view(l, Ws.shape[1] * Ws.shape[2]), 1, index.unsqueeze(1)).view(l, 1, 1)
        Wcols = torch.gather(Ws[need_updt], 2, t_idxs.view(-1,1,1).expand(-1,Ws.shape[1],-1))
        Wrows = torch.gather(Ws[need_updt], 1, t_mode_tgts.view(-1,1,1).expand(-1,-1,Ws.shape[2]))
        Wrows.scatter_add_(2, t_idxs.view(l, 1, 1), -torch.ones(l, dtype=real_type, device=device).view(l, 1, 1))
        Ws[need_updt] -= Wrows * Wcols / ratios
        ctx.need_grad = Ws.requires_grad
        if ctx.need_grad:
            ctx.need_updt, ctx.mode_tgts, ctx.idxs, ctx.ratios, ctx.Wcols, ctx.Wrows = need_updt, mode_tgts, idxs, ratios, Wcols, Wrows
        return Ws

    @staticmethod
    def backward(ctx, grad_Ws):
        l = len(ctx.need_updt)
        grad_Wouts = None
        if ctx.need_grad:
            grad_Wcols = ((grad_Ws * ctx.Wrows).sum(2).view(grad_Ws.shape[0], 1, grad_Ws.shape[1]) / ctx.ratios).view(grad_Ws.shape[0], grad_Ws.shape[1], 1)
            grad_Wrows = (grad_Wcols * grad_Ws).sum(1).view(grad_Ws.shape[0], 1, grad_Ws.shape[2]) / ctx.ratios
            grad_ratios = (grad_Wrows * ctx.Wrows).sum(2).unsqueeze(2) / ctx.ratios
            grad_Wouts = grad_Ws.clone()
            t_mode_tgts = torch.tensor(ctx.mode_tgts, dtype=int_type, device=device)
            t_idxs = torch.tensor(ctx.idxs, dtype=int_type, device=device)
            grad_Wouts[ctx.need_updt].scatter_add_(2, t_idxs.view(l, 1, 1).expand(-1, grad_Wouts.shape[1], 1), -grad_Wcols)
            grad_Wouts[ctx.need_updt].scatter_add_(1, t_mode_tgts.view(l, 1, 1).expand(-1, -1, grad_Wouts.shape[2]), -grad_Wrows)
            index = t_mode_tgts * grad_Wouts.shape[2] + t_idxs
            grad_Wouts[ctx.need_updt].view(l, grad_Wouts.shape[1] * grad_Wouts.shape[2]).scatter_add_(1, index.unsqueeze(1), grad_ratios.view(l, 1))
        return grad_Wouts, None, None, None
    
updateWs = UpdateWs.apply

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
    
    @torch.no_grad()
    def evaluate(self, state):
            result = torch.tensor(0., dtype=real_type, device=device)
            state.rise()
            for key, val in self.terms.items():
                try:
                    state1 = state.forward(key) # if key involes more than 1 hopping step, could raise state1.det -> nan
                    ratio = state1.sign * state.sign * (state1.logdet - state.logdet).exp()
                    ratio *= torch.exp((state.energy - state1.energy)/2)
                    result += ratio * val
                except ConfigurationError:
                    pass
            return result

    @torch.no_grad()
    def evaluates(self, states):
        results = torch.zeros(len(states.configs), dtype=real_type, device=device)
        states.rise()
        for key, val in self.terms.items():
            need_updt = states.scan(key)
            if need_updt != []:
                states1 = states.forward(need_updt, key)
                ratios = states1.signs[need_updt] * states.signs[need_updt] * (states1.logdets[need_updt] - states.logdets[need_updt]).exp()
                ratios *= torch.exp((states.energys[need_updt] - states1.energys[need_updt])/2)
                results.scatter_add_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios * val)
        return results
    
    @torch.no_grad()
    def mul_evaluate(self, states, bilinear=True):
        states.rise()
        n = len(states.configs)
        states1, vals = states.bilinear_replace(self) if bilinear else states.quartic_replace(self)
        ratios = states1.signs.view(len(self.terms), n) * states.signs * (states1.logdets.view(len(self.terms), n) - states.logdets).exp()
        ratios *= torch.exp((states.energys - states1.energys.view(len(self.terms), n))/2)
        results = (ratios * vals.view(len(self.terms), n)).sum(0)
        return results
    
    def chop(self, cut=None):
        ops = [self]
        new_ops = []
        if cut is not None:
            for _ in range(cut):
                for op in ops:
                    terms1 = dict(list(op.terms.items())[:len(op.terms)//2])
                    terms2 = dict(list(op.terms.items())[len(op.terms)//2:])
                    new_ops.append(type(self)(terms1))
                    new_ops.append(type(self)(terms2))
                ops = new_ops
                new_ops = []
        return ops 
    
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
    
    def qis(self, configs):
        qx = self.qs.unsqueeze(0).expand(len(configs), -1, -1)[0, configs]
        i = torch.tensor(configs, dtype=int_type, device=device).div(self.Nf, rounding_mode='floor')
        i = i.unsqueeze(-1).expand([-1, -1, qx.shape[-1]])
        qi = torch.zeros([len(configs), self.Ns, qx.shape[-1]], dtype=int_type, device=device)
        qi = qi.scatter_add(1, i, qx)
        return qi

    def qmsk(self, config, q0=None):
        dq = self.dq[:,config,:]
        if q0 is not None:
            dq = dq - q0.view(1,1,-1)
        qmsk = (dq==0).all(-1).to(dtype=real_type)
        return qmsk

    def qmsks(self, configs, q0s=None):
        idx = torch.tensor(configs, dtype=int_type, device=device).view(-1, 1, 2*self.Ns, 1).expand(-1, self.dq.shape[0], -1, 2)
        dqs = torch.gather(self.dq.unsqueeze(0).expand(len(configs), -1, -1, -1), 2, idx)
        if q0s is not None:
            dqs = dqs - q0s.view(len(configs), 1, 1, -1)
        qmsks = (dqs==0).all(-1).to(dtype=real_type)
        return qmsks
    
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

    def Ds(self, configs):
        return self.M.unsqueeze(0)[0, configs]
    
    def slogdets(self, configs):
        return torch.linalg.slogdet(self.Ds(configs))
    
    def Ws(self, configs):
        return torch.bmm(self.M.unsqueeze(0).expand(len(configs) ,-1 ,-1), torch.linalg.inv(self.Ds(configs)))
    
''' projector model '''
class ProjectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2, 3), dtype=real_type, device=device)) # total charge?

    def energy(self, qi):
        mu = - torch.nn.functional.log_softmax(self.weight, -1)
        idx = qi.abs()
        return sum(mu[i,idx[:,i]].sum() for i in range(2)) # i = 0
        
    def energys(self, qis):
        mu = - torch.nn.functional.log_softmax(self.weight, -1)
        idxs = qis.abs()
        #return sum(mu[i,idxs[:,:,i]].sum(-1) for i in range(2))
        return mu[0,idxs[:,:,0]].sum(-1)

''' variational model '''
class VariationalModel(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        self.lattice = HoneycombModel(L)
        self.basis = BasisSystem(self.lattice)
        self.meanfield = MeanfieldModel(self.lattice)
        self.projector = ProjectorModel()
        self._paras = torch.tensor([], dtype=real_type, device=device)
        self.reset()

    def reset(self):
        self._step = 0
        self._rejects = 0
        self.meanfield.reset()
        return self
    
    def record(self):
        self._paras = torch.cat((self._paras, torch.cat((self.meanfield.u.view(-1), self.projector.weight.view(-1))).unsqueeze(-1).clone().detach()), 1)
    
    def state(self, config=None):
        if config is None:
            config = self.basis.sample()
        return VariationalState(self, Configuration(config))
    
    def states(self, configs=None, wlks=1):
        legal = False
        if configs is None:
            while not legal:
                try:
                    configs = [Configuration(self.basis.sample()) for _ in range(wlks)]
                    sample_states = VariationalStates(self, configs)
                    for _ in range(5):
                        sample_states, logq = self.proposes1(sample_states)
                    legal = True
                except ConfigurationError:
                    pass
            configs = sample_states.configs
        return VariationalStates(self, configs)

    def propose1(self, state, flatten=False): # states
        trans_prob = state.trans_prob(0, flatten=flatten) # q0 = 0 on state0
        idx = torch.multinomial(trans_prob.sum(0), 1).item()
        mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
        mode_src = state.config[idx]
        if mode_tgt == mode_src:
            return state, torch.tensor(0., dtype=real_type, device=device)
        else:
            logq = trans_prob[mode_tgt, idx].log()
            state = state.clone() # state0 -> state1
            state.replace(mode_src, mode_tgt)
            trans_prob = state.trans_prob(0, flatten=flatten) # q0 = 0 on state1
            logq = logq - trans_prob[mode_src, idx].log()
            return state, logq

    def propose2(self, state, flatten=False): # states
        logq = torch.tensor(0., dtype=real_type, device=device)
        trans_prob = state.trans_prob(flatten=flatten) # q0 = any on state0
        idx = torch.multinomial(trans_prob.sum(0), 1).item()
        mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
        mode_src = state.config[idx]
        dq = self.basis.dq[mode_tgt, mode_src] # compute dq
        logq = logq + trans_prob[mode_tgt, idx].log()
        if mode_tgt != mode_src:
            state = state.clone() # state0 -> state1
            state.replace(mode_src, mode_tgt)
        trans_prob = state.trans_prob(-dq, flatten=flatten) # q0 = -dq on state1
        logq = logq - trans_prob[mode_src, idx].log()
        idx = torch.multinomial(trans_prob.sum(0), 1).item()
        mode_tgt = torch.multinomial(trans_prob[:,idx], 1).item()
        mode_src = state.config[idx]
        logq = logq + trans_prob[mode_tgt, idx].log()
        if mode_tgt != mode_src:
            state = state.clone() # state1 -> state2
            state.replace(mode_src, mode_tgt)
        trans_prob = state.trans_prob(flatten=flatten) # q0 = any on state2
        logq = logq - trans_prob[mode_src, idx].log()
        return state, logq

    def MCstep(self, state, flatten=False): # states
        self._step += 1
        if self._step % 5 == 0:
            propose = self.propose2
        else:
            propose = self.propose1
        new_state, logq = propose(state, flatten=flatten)
        new_logprob = new_state.logprob
        old_logprob = state.logprob
        logq = new_logprob.detach() - old_logprob.detach() if flatten else new_logprob.detach() - old_logprob.detach() - logq
        if torch.exp(logq) > random.random():
            return new_state, new_logprob
        else:
            self._rejects += 1
            return state, old_logprob
        
    def MCrun(self, H=None, state=None, steps=1, flatten=False): # states
        if state is None: # no state provided, sample a random one
            state = self.state()
        Hval = torch.tensor(0., dtype=real_type, device=device)
        ls = []
        logprobs = []
        for _ in range(steps):
            state, logprob = self.MCstep(state, flatten=flatten)
            if H is not None:
                Heva = H.evaluate(state)
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
    
    def proposes1(self, states, carry_prob=True, flatten=False): # states
        trans_probs = states.trans_probs(0, flatten=flatten) # q0s = 0 on states0
        idxs = torch.multinomial(trans_probs.sum(1), 1)
        mode_tgts = torch.multinomial(torch.gather(trans_probs, 2, idxs.unsqueeze(1).expand(-1, trans_probs.shape[1], -1)).view(trans_probs.shape[0], trans_probs.shape[1]), 1).view(trans_probs.shape[0]).tolist()
        mode_srcs = torch.gather(torch.tensor(states.configs, dtype=int_type, device=device), 1, idxs).view(trans_probs.shape[0]).tolist()
        if mode_tgts is mode_srcs:
            if carry_prob:
                return states, torch.zeros(trans_probs.shape[0], dtype=real_type, device=device)
            else:
                return states
        else:
            if carry_prob:
                index1 = torch.tensor(mode_tgts, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
                logqs = torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index1).log()
                states = states.clone() # state0 -> state1
            states.replace(mode_srcs, mode_tgts)
            if carry_prob:
                trans_probs = states.trans_probs(0, flatten=flatten) # q0 = 0 on state1
                index2 = torch.tensor(mode_srcs, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
                logqs = logqs - torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index2).log()
                return states, logqs.view(trans_probs.shape[0])
            else:
                return states
    
    def proposes2(self, states, carry_prob=True, flatten=False): # states
        trans_probs = states.trans_probs(flatten=flatten) # q0 = any on state0
        if carry_prob:
            logqs = torch.zeros(trans_probs.shape[0], dtype=real_type, device=device).view(trans_probs.shape[0], 1)
        idxs = torch.multinomial(trans_probs.sum(1), 1)
        mode_tgts = torch.multinomial(torch.gather(trans_probs, 2, idxs.unsqueeze(1).expand(-1, trans_probs.shape[1], -1)).view(trans_probs.shape[0], trans_probs.shape[1]), 1).view(trans_probs.shape[0]).tolist()
        mode_srcs = torch.gather(torch.tensor(states.configs, dtype=int_type, device=device), 1, idxs).view(trans_probs.shape[0]).tolist()
        dqs = self.basis.dq[mode_tgts, mode_srcs] # compute dqs
        if carry_prob:
            index1 = torch.tensor(mode_tgts, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
            logqs = logqs + torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index1).log()
        if mode_tgts is not mode_srcs:
            if carry_prob:
                states = states.clone() # state0 -> state1
            states.replace(mode_srcs, mode_tgts)
        trans_probs = states.trans_probs(-dqs, flatten=flatten) # q0 = -dq on state1
        if carry_prob:
            index2 = torch.tensor(mode_srcs, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
            logqs = logqs - torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index2).log()
        idxs = torch.multinomial(trans_probs.sum(1), 1)
        mode_tgts = torch.multinomial(torch.gather(trans_probs, 2, idxs.unsqueeze(1).expand(-1, trans_probs.shape[1], -1)).view(trans_probs.shape[0], trans_probs.shape[1]), 1).view(trans_probs.shape[0]).tolist()
        mode_srcs = torch.gather(torch.tensor(states.configs, dtype=int_type, device=device), 1, idxs).view(trans_probs.shape[0]).tolist()
        if carry_prob:
            index1 = torch.tensor(mode_tgts, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
            logqs = logqs + torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index1).log()
        if mode_tgts is not mode_srcs:
            if carry_prob:
                states = states.clone() # state1 -> state2
            states.replace(mode_srcs, mode_tgts)
        if carry_prob:
            trans_probs = states.trans_probs(flatten=flatten) # q0 = any on state2
            index2 = torch.tensor(mode_srcs, dtype=int_type, device=device).unsqueeze(1) * trans_probs.shape[2] + idxs
            logqs = logqs - torch.gather(trans_probs.view(trans_probs.shape[0], trans_probs.shape[1] * trans_probs.shape[2]), 1, index2).log()
            return states, logqs.view(trans_probs.shape[0])
        else:
            return states

    def MCsteps(self, states, carry_prob=True, flatten=False): # states
        self._step += 1
        if self._step % 5 == 0:
            proposes = self.proposes2
        else:
            proposes = self.proposes1
        if carry_prob:
            new_states, logqs = proposes(states, carry_prob=carry_prob, flatten=flatten)
            new_logprobs = new_states.logprobs
            old_logprobs = states.logprobs
            logqs = new_logprobs.detach() - old_logprobs.detach() if flatten else new_logprobs.detach() - old_logprobs.detach() - logqs
            gate = torch.exp(logqs) > torch.rand(logqs.shape[0], device=device)
            new_states1 = states.merge(new_states, gate)
            self._rejects += (~gate).long().sum().item()/len(states.configs)
            return new_states1, new_states1.logprobs
        else:
            new_states = proposes(states)
            return new_states
        
    def MCruns(self, H=None, cut=(None, None), states=None, wlks=4, steps=1, carry_prob=True, flatten=False): # states
        if states is None:
            states = self.states(wlks=wlks) # no states provided, sample a random one
        else:
            wlks = len(states.configs)
        configs = []
        if carry_prob:
            for _ in range(steps):
                states, logprobs = self.MCsteps(states, carry_prob=carry_prob, flatten=flatten)
                configs += states.configs
        else:
            for _ in range(steps):
                states = self.MCsteps(states, carry_prob=carry_prob, flatten=True) # uniformlly sample needs to flatten trans_prob matrix
                configs += states.configs
        if H is None:
            return states
        else:
            states_all = self.states(configs)
            logprobs = states_all.logprobs
            states_all.rise()
            (Ht, HJ) = H
            (cutt, cutJ) = cut
            Ht, HJ = Ht.chop(cutt), HJ.chop(cutJ)
            Hvals = sum(ht.mul_evaluate(states_all, bilinear=True) for ht in Ht) + sum(hJ.mul_evaluate(states_all, bilinear=False) for hJ in HJ)
            MCeva = Hvals.sum()/(steps*wlks)
            obj = (Hvals - MCeva) @ logprobs
            return obj/(steps*wlks), MCeva, states 
        
    def show(self, iteration=None):
        if iteration is None:
            iteration = self._paras.shape[1]
        x = numpy.linspace(1, iteration, iteration)
        parameters = self._paras.cpu().numpy()
        for i in range(parameters.shape[0]):
            pyplot.plot(x, parameters[:, parameters.shape[1]-iteration:][i], linewidth=2.0)
        pyplot.xlabel("iteration")
        pyplot.ylabel("parameters")
        return pyplot.show()
        
''' variational states'''
class VariationalStates():
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.reset()
        
    def __repr__(self):
        return type(self).__name__ + '({})'.format(self.configs)   
        
    def reset(self):
        self._signs = None
        self._logdets = None
        self._Ws = None
        self._temps = None
        self._energys = None
        return self

    def clone(self):
        configs = [config.clone() for config in self.configs]
        new = type(self)(self.model, configs)
        # clone/link any cache variables that is not None
        if self._signs is not None:
            new._signs = self._signs.clone()
        if self._logdets is not None:
            new._logdets = self._logdets.clone()
        if self._Ws is not None:
            new._Ws = self._Ws.clone()
        if self._temps is not None:
            new._temps = self._temps
        if self._energys is not None:
            new._energys = self._energys # no need to clone
        return new 
    
    # clone n copies of states
    def clones(self, n=1, shell=False): # shell returns a VariationalStates without Ws & energys
        configs = []
        configs = [config.clone() for _ in range(n) for config in self.configs]
        new = type(self)(self.model, configs)
        # clone/link any cache variables that is not None
        if self._signs is not None:
            new._signs = self._signs.repeat(n).clone()
        if self._logdets is not None:
            new._logdets = self._logdets.repeat(n).clone()
        if not shell:
            if self._Ws is not None:
                new._Ws = self._Ws.repeat(n, 1, 1).clone()
            if self._temps is not None:
                old_temps = self._temps
                need_updt = []
                for _ in range(n):
                    need_updt += [i + len(self.configs) * _ for i in old_temps[0]]
                new._temps = (need_updt, old_temps[1] * n, old_temps[2] * n)
            #if self._energys is not None:
                #new._energys = self._energys.repeat(n) # energy has to be recalculate
        return new 
    
    # merge two states based on gate(tensor of T or F)
    def merge(self, other, gate):
        self.rise()
        other.rise() # clear temps
        configs = []
        for i in range(gate.shape[0]):
            if gate[i]:
                configs.append(other.configs[i].clone())
            else:
                configs.append(self.configs[i].clone())
        new = type(self)(self.model, configs)
        gate1 = gate.long()
        gate2 = (~gate).long()
        new._signs = other._signs.clone() * gate1 + self._signs.clone() * gate2
        new._logdets = other._logdets.clone() * gate1 + self._logdets.clone() * gate2
        new._Ws = other._Ws.clone() * gate1.view(gate.shape[0], 1, 1) + self._Ws.clone() * gate2.view(gate.shape[0], 1, 1)
        new._energys = other._energys * gate1 + self._energys * gate2
        return new

    def rise(self):
        self.logprobs
        self.Ws
    
    def split(self):
        statelist = []
        for config in self.configs:
            statelist.append(VariationalState(self.model, config))
        return statelist
    
    @property
    def signs(self):
        if self._signs is None:
            self._signs, self._logdets = self.model.meanfield.slogdets(self.configs)
        return self._signs
    
    @property
    def logdets(self):
        if self._logdets is None:
            self._signs, self._logdets = self.model.meanfield.slogdets(self.configs)
        return self._logdets
    
    @property
    def dets(self):
        return self.signs * self.logdets.exp()

    @property
    def qis(self):
        return self.model.basis.qis(self.configs)
    
    @property
    def energys(self):
        if self._energys is None:
            self._energys = self.model.projector.energys(self.qis)
        return self._energys
    
    @property
    def logprobs(self):
        return 2*self.logdets - self.energys
    
    @property
    def Ws(self):
        if self._Ws is None: # W has not been initialized
            self._Ws = self.model.meanfield.Ws(self.configs)
        if self._temps is not None: # W has not been updated
            self._Ws = updateWs(self._Ws, *self._temps) # update W
            self._temps = None # clear temp
        return self._Ws

    # need_updt, mode_srcs & mode_tgts are lists
    def replace(self, mode_srcs, mode_tgts, need_updt=None):
        if need_updt is None:
            need_updt = list(range(len(self.configs)))
        elif need_updt == []:
            return self
        idxs = []
        for i, mode_src in zip(need_updt, mode_srcs):
            try:
                idxs.append(self.configs[i].modeidx[mode_src])
            except KeyError:
                raise ConfigurationError(action='annihilated', mode=mode_src) from None
        if not (mode_srcs is mode_tgts):
            index = torch.tensor(mode_tgts, dtype=int_type, device=device) * self.Ws.shape[2] + torch.tensor(idxs, dtype=int_type, device=device)
            ratios = torch.gather(self.Ws[need_updt].view(len(need_updt), self.Ws.shape[1] * self.Ws.shape[2]), 1, index.unsqueeze(1)).view(-1)
            self._signs = self.signs.scatter_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios.sign(), reduce='multiply')
            self._logdets = self.logdets.scatter_add_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios.abs().log())
            # config should NOT be modified before evaluation of slogdet
            for i, idx, mode_tgt in zip(need_updt, idxs, mode_tgts): # update configurations
                self.configs[i][idx] = mode_tgt
            self._temps = (need_updt, mode_tgts, idxs) # cache temp data
            self._energys = None # energy should be recalculated
        return self
    
    # brutal replace by bilinear operator
    @torch.no_grad()
    def bilinear_replace(self, op):
        need_updt = []
        mode_srcs = []
        mode_tgts = []
        idxs = []
        new_states = self.clones(n=len(op.terms), shell=True) # get a shell states with n copies
        vals = torch.zeros(len(op.terms) * len(self.configs), dtype=real_type, device=device)
        i = 0
        for key, val in op.terms.items():
            ((mode_src, mode_tgt),) = key
            for n in range(len(self.configs)):
                config_idx = i * len(self.configs) + n
                try:
                    new_states.configs[config_idx].replace(mode_src, mode_tgt) # update configuration
                    need_updt.append(config_idx)
                    mode_srcs.append(mode_src)
                    mode_tgts.append(mode_tgt)
                    vals[config_idx] += val
                    idxs.append(new_states.configs[config_idx].modeidx[mode_tgt])
                except ConfigurationError:
                    new_states.configs[config_idx] = self.configs[n] # return to the original config
                    pass
            i += 1
        index = torch.tensor(mode_tgts, dtype=int_type, device=device) * self.Ws.shape[2] + torch.tensor(idxs, dtype=int_type, device=device)
        Ws_idx = (torch.tensor(need_updt, dtype=int_type, device=device) % len(self.configs)).tolist()
        ratios = torch.gather(self.Ws[Ws_idx].view(len(need_updt), self.Ws.shape[1] * self.Ws.shape[2]), 1, index.unsqueeze(1)).view(-1)
        new_states._signs = new_states.signs.scatter_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios.sign(), reduce='multiply')
        new_states._logdets = new_states.logdets.scatter_add_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios.abs().log())
        return new_states, vals
            
    # brutal replace by quartic operator
    @torch.no_grad()
    def quartic_replace(self, op):
        need_updt = []
        mode_srcs1 = []
        mode_tgts1 = []
        mode_srcs2 = []
        mode_tgts2 = []
        idxs1 = []
        idxs2 = []
        new_states = self.clones(n=len(op.terms), shell=False) # need to store Ws
        vals = torch.zeros(len(op.terms) * len(self.configs), dtype=real_type, device=device)
        i = 0
        for key, val in op.terms.items():
            ((mode_src1, mode_tgt1), (mode_src2, mode_tgt2)) = key
            for n in range(len(self.configs)):
                config_idx = i * len(self.configs) + n
                try:
                    new_states.configs[config_idx].replace(mode_src1, mode_tgt1) # update configuration
                    new_states.configs[config_idx].replace(mode_src2, mode_tgt2)
                    idxs1.append(self.configs[n].modeidx[mode_src1])
                    idxs2.append(self.configs[n].modeidx[mode_src2])
                    need_updt.append(config_idx)
                    mode_srcs1.append(mode_src1)
                    mode_tgts1.append(mode_tgt1)
                    mode_srcs2.append(mode_src2)
                    mode_tgts2.append(mode_tgt2)
                    vals[config_idx] += val
                except ConfigurationError:
                    new_states.configs[config_idx] = self.configs[n] # return to the original config
                    pass
            i += 1
        index1 = torch.tensor(mode_tgts1, dtype=int_type, device=device) * self.Ws.shape[2] + torch.tensor(idxs1, dtype=int_type, device=device)
        ratios1 = torch.gather(new_states.Ws[need_updt].view(len(need_updt), self.Ws.shape[1] * self.Ws.shape[2]), 1, index1.unsqueeze(1)).view(-1)
        new_states._signs = new_states.signs.scatter_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios1.sign(), reduce='multiply')
        new_states._logdets = new_states.logdets.scatter_add_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios1.abs().log())
        new_states._Ws = updateWs(new_states._Ws, *(need_updt, mode_tgts1, idxs1))
        index2 = torch.tensor(mode_tgts2, dtype=int_type, device=device) * self.Ws.shape[2] + torch.tensor(idxs2, dtype=int_type, device=device)
        ratios2 = torch.gather(new_states.Ws[need_updt].view(len(need_updt), self.Ws.shape[1] * self.Ws.shape[2]), 1, index2.unsqueeze(1)).view(-1)
        new_states._Ws = None
        torch.cuda.empty_cache()
        new_states._signs = new_states.signs.scatter_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios2.sign(), reduce='multiply')
        new_states._logdets = new_states.logdets.scatter_add_(0, torch.tensor(need_updt, dtype=int_type, device=device), ratios2.abs().log())     
        return new_states, vals
        
    def scan(self, instruction):
        need_updt = []
        for i in range(len(self.configs)):
            config = self.configs[i].clone()
            try:
                for mode_src, mode_tgt in instruction:
                    config.replace(mode_src, mode_tgt)
                need_updt += [i]
            except ConfigurationError:
                pass
        return need_updt
    
    def forward(self, need_updt, instruction):
        if need_updt == []:
            return self
        new = self.clone()
        for mode_src, mode_tgt in instruction:
            mode_srcs, mode_tgts = [mode_src] * len(need_updt), [mode_tgt] * len(need_updt)
            try:
                new.replace(mode_srcs, mode_tgts, need_updt)
            except ConfigurationError as err:
                raise err from None
        return new

    def backward(self, need_updt, instruction):
        if need_updt == []:
            return self
        new = self.clone()
        for mode_src, mode_tgt in reversed(instruction):
            mode_srcs, mode_tgts = [mode_src] * len(need_updt), [mode_tgt] * len(need_updt)
            try:
                new.replace(mode_tgts, mode_srcs, need_updt)
            except ConfigurationError as err:
                raise err from None
        return new
    
    def trans_probs(self, q0s=None, flatten=False):
        weight = self.Ws.detach()**2
        if q0s is not None:
            if q0s is 0:
                weight = weight * self.model.basis.qmsks(self.configs)
            else:
                weight = weight * self.model.basis.qmsks(self.configs, q0s)
        trans_probs = weight / weight.view(-1, 1, 8 * self.model.lattice.Ns ** 2).sum(2).unsqueeze(1)
        if flatten:
            d = trans_probs.max().log10().round().abs() + 10
            trans_probs = trans_probs.round(decimals = int(d)).ceil()
        return trans_probs

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

    def rise(self):
        self.logprob
        self.W
    
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

    def trans_prob(self, q0=None, flatten=False):
        weight = self.W.detach()**2
        if q0 is not None:
            if q0 is 0:
                weight = weight * self.model.basis.qmsk(self.config)
            else:
                weight = weight * self.model.basis.qmsk(self.config, q0)
        trans_prob = weight / weight.sum()
        if flatten:
            d = trans_prob.max().log10().round().abs() + 10
            trans_prob = trans_prob.round(decimals = int(d)).ceil()
        return trans_prob
    
'''optimization'''
    
def optm(states, L=9, lr_decay=True, J=1.0, lr=0.1, gamma=0.95, itr=101, cut=(None, None), load=None, save=None):
    mdl = VariationalModel(L)
    if load is not None:
        mdl.load_state_dict(torch.load(load))
        mdl.eval()
    states.model = mdl
    H = (-1 * mdl.lattice.Ht, J * mdl.lattice.HJ)
    optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
    if lr_decay:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    for epoch in range(itr):
        #for _ in range(100):
            #states = mdl.MCruns(states=states, steps=20, carry_prob=True, flatten=False).reset()
        mdl.reset()
        obj, MCenergy, states = mdl.MCruns(H=H, cut=cut, states=states, steps=20, carry_prob=True, flatten=False)
        loss = obj
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_decay:
            scheduler.step()
        if epoch % 20 == 0 and epoch > 5:
            mdl.show()
        if epoch % 5 == 0:
            print('MCenergy =', MCenergy.item(), 'rejection rate =',mdl._rejects/mdl._step)
        mdl.reset()
        states.reset()
        mdl.record()
        if save is not None:
            torch.save(mdl.state_dict(), save)
    return mdl
    
def heat(L=9, wlks=5, itr=1001, load=None):
    mdl = VariationalModel(L)
    states = mdl.MCruns(steps=20, wlks=wlks, carry_prob=True, flatten=False).reset()
    for _ in range(itr):
        mdl.reset()
        states = mdl.MCruns(states=states, steps=20, carry_prob=True, flatten=False).reset()
        if _%(itr//4)==0:
            print(mdl._rejects/mdl._step, _)
    if load is None:
        return states
    else:
        mdl.load_state_dict(torch.load(load))
        mdl.eval()
        states.model = mdl
        for _ in range(itr):
            mdl.reset()
            states = mdl.MCruns(states=states, steps=20, carry_prob=True, flatten=False).reset()
            if _%(itr//4)==0:
                print(mdl._rejects/mdl._step, _)
        return states
    
'''sanity check for L=1'''    

def test(mdl):
    H = -0.2 * mdl.lattice.Ht + mdl.lattice.HJ
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
        
        