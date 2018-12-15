#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import fsolve, bisect
from itertools import product
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
import plotly.io as pio
import plotly.graph_objs as go
init_notebook_mode()
import os


# # Complete Market: Replication

# In this part we try to replicate some results of Lucas Stokey (1983)

# We assume that the utility function is of the form:
# 
# $$U(c,n) = \frac{c^{1-\rho} - 1}{1-\rho} - \chi \frac{n^{1+\psi}}{1+\psi}$$
# 
# where $\chi=1.5$, $\psi=1$ and $\rho=1.01$.
# 
# We also assume that government spending follow a two-state Markov chain, with $g_L$ =0.1, $g_H$ =0.2,
# $g_0$ = $g_L$, and the transition matrix $\Pi$=$
# \left(\begin{array}{cc} 
# 0.9 & 0.1\\
# 0.5 & 0.5
# \end{array}\right)$

# In[2]:


# global variables (not pythonic but in the spirit of the exercise)
chi = 1.5
psi = 1
rho = 1.01
beta = .9
g = np.array([.1, .2])
pi = np.array([[.9, .1], [.5, .5]])


# In[3]:


plot_path = '../plots' 
plotnb = 1


# ## Q1. $t \geq 1$

# ### a. Plot the history-independent and time-invariant tax $\tau(\Phi)$

# From the lecture notes we know that $$\tau = \frac{\Phi(\rho + \psi)}{1 + \Phi(1+\psi)}$$

# In[4]:


def tau(phi: float, rho: float = rho, psi: float = psi):
    """Returns the the tax rate"""
    return phi*(rho + psi) / (1 + phi*(1 + psi))


# In[5]:


x = np.arange(0, 1, .01)
y = tau(x)
data = [go.Scatter(x=x, y=y, mode="lines")]
layout = go.Layout(title='$Tax~rate~as~a~function~of~\Phi$',
                   xaxis=dict(title='$\Phi$'),
                   yaxis=dict(title='$\\tau(\Phi)$'))
fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)


# ### b. Plot the history-independent and time-invariant allocations ${c(\Phi,g), n(\Phi, g)} \forall g$, as a function of $\Phi$.
# 

# From the lecture notes, using the $*$ equation and the feasibility constraint we know that the following relations hold:
# 
# $$c(g) + g - n(g) = 0$$
# $$(1 + \Phi) (u_c(g) + u_n(g)) + \Phi (c(g) u_{cc}(g) + n(g) u_{nn}(g)) = 0$$

# In[6]:


def star(c, n, phi, b0):
    """Returns the star equation """
    u_c = c**(-rho)
    u_cc = -rho * c**(-rho - 1)
    u_n = -chi * n**psi
    u_nn = -chi * psi * n**(psi - 1)
    return (1 + phi) * (u_c + u_n) + phi * ((c-b0) * u_cc + n * u_nn)


# In[7]:


def allocation(z, *args):
    """Returns the allocation system"""
    c, n = z
    g, phi, b0 = args
    return (c + g - n,
            star(c, n, phi, b0))


# In[8]:


b0 = 0
args_list = [args_ for args_ in product(g, np.arange(0, 1, .01))]
args_list = [(*args_list[i], b0) for i in range(len(args_list))]
z = [tuple(fsolve(allocation, (.42, .42), args))
     for args in args_list]
data_ = [[*args_list[i], *z[i]]
         for i in range(len(args_list))]
df_cn = pd.DataFrame(
    data_, columns=['state', 'phi', 'b0', 'Consumption', 'Labor supply'])


# In[9]:


for v in ['Consumption', 'Labor supply']:
    data = []
    for s in g:
        tmp = df_cn.query('state==@s')
        data += [go.Scatter(
            x=tmp['phi'],
            y=tmp[v],
            name='g={}'.format(s)
        )]
    layout = go.Layout(title='${}~in~function~of~(\Phi, g)$'.format(v),
                       xaxis=dict(title='$\Phi$'),
                       yaxis=dict(title='${}(\Phi)$'.format(v))
                       )
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
    plotnb += 1
    iplot(fig)


# ### c. Write the implementability constraints recursively to find ${b(\Phi, g)}$, and plot debt policies as a function of $\Phi$.

# From the lecture notes, using the implementability constraint, we know that the following relations hold:
# 
# $$u_{c_L}c_L + u_{n_L}n_L + \beta (\pi_{LL}u_{c_L}b_L + \pi_{LH}u_{c_H}b_H) -u_{c_L}b_L = 0$$
# $$u_{c_H}c_H + u_{n_H}n_H + \beta (\pi_{HL}u_{c_L}b_L + \pi_{HH}u_{c_H}b_H)-u_{c_H}b_H=0$$

# In[10]:


def bonds(z, *args):
    """Returns bonds level"""
    bl, bh = z
    phi = args
    cl, nl = df_cn.query('phi==@phi and state==.1')['Consumption'].values[0], df_cn.query(
        'phi==@phi and state==.1')['Labor supply'].values[0]
    ch, nh = df_cn.query('phi==@phi and state==.2')['Consumption'].values[0], df_cn.query(
        'phi==@phi and state==.2')['Labor supply'].values[0]

    def u_c(c):
        return c**(-rho)

    def u_n(n):
        return -chi * n**psi

    return (u_c(cl) * cl + u_n(nl) * nl + beta *
            (pi[0, 0] * u_c(cl) * bl + pi[0, 1] * u_c(ch) * bh) - u_c(cl) * bl,
            u_c(ch) * ch + u_n(nh) * nh + beta *
            (pi[1, 0] * u_c(cl) * bl + pi[1, 1] * u_c(ch) * bh) - u_c(ch) * bh)


# In[11]:


z = [tuple(fsolve(bonds, (.42, .42), phi))
     for phi in np.arange(0, 1, .01)]
df_b = pd.DataFrame([(i / 100, *z[i])
                     for i in range(len(z))],
                    columns=['phi', 'low_state', 'high_state'])


# In[12]:


data = [
    go.Scatter(x=df_b['phi'], y=df_b['low_state'], name='g=0.1'),
    go.Scatter(x=df_b['phi'], y=df_b['high_state'], name='g=0.2')
]
layout = go.Layout(
    title='$Bonds~in~function~of~(\Phi, g)$'.format(v),
    xaxis=dict(title='$\Phi$'),
    yaxis=dict(title='$b(\Phi)$'))
fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)


# ### d. What can you say about the relationship between $\Phi$ and $\tau$? Between $\Phi$ and $c$?

# $\Phi$ is the Lagrange multiplier associated to the implementability constraint in the Ramsey plan. It can be interpreted as a measure of the shadow cost of the distortion implied by the labor tax rate. In this context, a positive relationship between $\Phi$ and $\tau$ is expected.
# 
# On the contrary, the relation between $\Phi$ and $c$ appears to be negative. This is due to the substitution effect caused by the labor tax rate. As $\tau$ (and thus $\Phi$) rises, agents lower their labor supply and hence earn lower revenues which negatively affects their consumption level. 

# ## t=0

# From the lecture notes, using the feasibility constraint at $t=0$ and the ($*-0$) equation, we know that the following relations hold:
# 
# $$c_0 + g_0 = n_0$$
# $$(1 + \Phi) (u_{c,0} + u_{n,0}) + \Phi ((c_0 -b_0) u_{cc,0} + n_0 u_{nn,0}) = 0$$

# In[13]:


ngrid = 100
g_ = .1
z_c = np.empty([ngrid, ngrid])
z_n = np.empty([ngrid, ngrid])
phi_ = np.linspace(0, 1, ngrid)
b0_ = np.linspace(-.1, .1, ngrid)
for i in range(100):
    for j in range(100):
        z_c[i, j], z_n[i, j] = fsolve(allocation, (.5, .5),
                                      (g_, phi_[i], b0_[j]))


# In[14]:


for v in ['Consumption', 'Labor supply']:
    if v == 'Consumption':
        z_tmp = z_c
    else:
        z_tmp = z_n
    data = [
        go.Surface(
            z=z_tmp,
            x=np.linspace(0, 1, 100),
            y=np.linspace(-.1, .1, 100),
        )
    ]
    layout = go.Layout(
        title='${}~at~time~0~in~function~of~(\Phi, b_0)$'.format(v),
        autosize=True,
        xaxis=dict(title='$\Phi$'),
        yaxis=dict(title='$b$'),
        #    zaxis=dict(title='$Consumption_0$')
    )
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
    plotnb += 1
    iplot(fig)


# ### b. Use a bisection method to find $\Phi$ as a function of $b_0$. Plot $\Phi$ as a function of $b_0$.

# In[15]:


def implementability(phi, *args):
    b0, g = args
    args = (g, phi, b0)
    c, n = fsolve(allocation, (0.1, 0.1), args)
    u_c = c**(-rho)
    u_n = -chi * n**psi
    return u_c*c + u_n * n - u_c * b0


# In[16]:


x = np.linspace(-.099, .099, 100)
y = [bisect(implementability, 0, 1, (b0, .1)) for b0 in x]


# In[17]:


data = [go.Scatter(x=x, y=y)]
layout = go.Layout(title='$\Phi~as~a~function~of~b_0$',
                   xaxis=dict(title='$b_0$'),
                   yaxis=dict(title='$\Phi(b_0)$'))
fig = go.Figure(data=data, layout=layout)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)


# ### c. What can you say about the relation between $b_0$ and $\Phi$?

# The initial debt level $b_0$ and $\Phi$ are positively related. Keeping in mind that $\Phi$ can be interpreted as a measure of the distortion caused by the labor tax, we understand that this relation comes from the need to levy distortionary taxes to repay the initial debt. 
# 
# It is also interesting to note that the distortion becomes null ($\Phi(b_0)$) when the government can fully finance public spendings with its initial endowment (when $-b_0 = g_0 = 0.1$).

# ## Simulation

# Simulate the economy for 100 periods. Plot the simulated govern- ment spending, allocations, tax and debt-to-output ratios. 

# In[18]:


def markovseq(length: int, states, transition: np.matrix, seed=42):
    """Simulates a Markov sequence"""
    np.random.seed(seed)
    seq = [states[0]]
    for i in range(length):
        if seq[-1] == states[0]:  # low state
            seq += [np.random.choice(states, p=pi[0])]
        else:  # high state
            seq += [np.random.choice(states, p=pi[1])]
    return seq


# In[19]:


def simul(sequence, b0: float):
    """Simulates the economy"""
    # init
    g_ = sequence[0]
    g = [sequence[0]]
    b = [b0]
    phi_ = bisect(implementability, 0, 1, (b0, g))
    args = (g_, phi_, b0)
    c_, n_ = fsolve(allocation, (.42, .42), args)
    c, n = [c_], [n_]
    t = [(c_**(-rho) - chi * n_**psi)/(c_**(-rho))]

    # iter
    phi_ = round(phi_, 2)
    for i in np.arange(1, len(sequence)):
        state_ = sequence[i]
        g += [state_]
        args = (g[-1], phi_, b[-1])
        c_, n_ = fsolve(allocation, (.42, .42), args)
        c += [c_]
        n += [n_]
        t += [tau(phi=phi_)]
        if state_ == .1:
            b += [df_b.query('phi==@phi_')['low_state'].values[0]]
        else:
            b += [df_b.query('phi==@phi_')['high_state'].values[0]]

    return g, b, c, n, t


# In[20]:


def plot_simul(simul):
    """Plots the simulated economy"""
    g, b, c, n, t = simul

    def quadrant(series, **kwargs):
        """"""
        periods = np.arange(len(series))
        trace = go.Scatter(x=periods, y=series,
                           mode='lines', marker=dict(color='rgb(145,191,219)'),
                           **kwargs)
        return trace

    c_trace = quadrant(c)
    n_trace = quadrant(n)
    b_trace = quadrant(b)
    g_trace = quadrant(g)
    t_trace = quadrant(t)
    bn_trace = quadrant(np.divide(b, n))

    fig = tools.make_subplots(rows=3, cols=2,
                              subplot_titles=('Consumption',
                                              'Labor supply',
                                              'Government debt',
                                              'Government expenditures',
                                              'Tax rate',
                                              'Debt/output ratio'))
    fig.append_trace(c_trace, 1, 1)
    fig.append_trace(n_trace, 1, 2)
    fig.append_trace(b_trace, 2, 1)
    fig.append_trace(g_trace, 2, 2)
    fig.append_trace(t_trace, 3, 1)
    fig.append_trace(bn_trace, 3, 2)

    fig.layout.update(title='$Simulated~economy~under~b_0={}$'.format(b[0]),
                      showlegend=False)

    return fig


# In[21]:


seq = markovseq(100, (.1, .2), pi)


# ### a. Assume $b_0 = 0$. What can you say about τ0?

# The tax rate is constant. This is due to the fact that if $b_0=0$, ten all periods are the same ($* = *-0$).

# In[22]:


simul_ = simul(seq, 0)
fig = plot_simul(simul_)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)


# ### Now assume b0 = 0.1. 

# In[23]:


simul_ = simul(seq, .1)
fig = plot_simul(simul_)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)


# ### Now assume b0 = −0.1.

# In[24]:


simul_ = simul(seq, -.099)
fig = plot_simul(simul_)
pio.write_image(fig, plot_path + '/fig{}.pdf'.format(plotnb))
plotnb += 1
iplot(fig)

