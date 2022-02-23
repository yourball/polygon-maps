import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
# import altair as alt
import time
import matplotlib.pyplot as plt

def force(x, k_list, xi_list, d):
    # precompute parameters of the force function
    ai_list, bi_list = [], []
    for indx in range(len(k_list)):
        bi = 0
        for i in range(1, indx):
            bi += k_list[i] * (xi_list[i] - xi_list[i - 1])
        if indx > 1:
            ai = xi_list[indx - 1]
        else:
            ai = 0
        ai_list.append(ai)
        bi_list.append(bi)
    ai_list, bi_list = np.array(ai_list), np.array(bi_list)

    intervals = xi_list < x
    indx = np.count_nonzero(intervals)
    ki = k_list[indx]
    ai, bi = ai_list[indx], bi_list[indx]
    f = ki * (x - ai) + bi + d
    return f

def orbit(
    x0, y0, k_list, xi_list, d, Tmax
):
    x, y = x0, y0
    traj_list = [[x0, y0]]
    for iter in range(Tmax):
        x, y = y, -x + force(y, k_list, xi_list, d)
        # traj_data = np.vstack([traj_data, [x, y]])
        traj_list += [[x, y]]
    traj_data = np.vstack(traj_list)
    return traj_data

def plot_orbits(ax, k_list, xi_list, d, Tmax=1000):
    x0_list = np.linspace(0, 10, 50)
    max_x0 = max(x0_list)
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(x0_list)
    for i, x0 in enumerate(x0_list):
        y0 = x0*1.01
        traj_data = orbit(x0, y0, k_list, xi_list, d, Tmax=Tmax)

        ax.scatter(traj_data[:, 0], traj_data[:, 1], s=0.25,
                   color=cm(1.*i/NUM_COLORS)
                   #color=[abs(x0)/max_x0, .5*abs(x0)/max_x0, 1-abs(x0)/max_x0]
                   )
    return ax

st.markdown("""## Canonical form of symplectic maps""")
st.write("""Arbitrary area-preserving (symplectic) mapping of the plane
can be represented in the McMillan-Turaev canonical form:""")
st.latex(r"""
\begin{cases}
q_{n+1} = F(q_n ,p_n)\\
p_{n+1} = G(q_n ,p_n)
\end{cases}
\quad
\Rightarrow
\quad
\begin{cases}
q_{n+1} = p_n\\
p_{n+1} = -q_n + f(p_n),
\end{cases}
""")
st.write("""where f is a force function.""")

k_list = []

st.write("""
        """)

# col1, col2 = st.columns(2)

num_pieces = st.sidebar.number_input('Number of piecewise regions for the force function f(q)',
                             min_value=2, max_value=10, value=3, step=1)
Tmax = st.sidebar.number_input("Number of map iterations", min_value=1, value=2000, step=1000)

st.sidebar.write(r'Specify slopes of the piesewise force function $k_i$:')

k_init = (1 + np.arange(num_pieces)) % 2
for p in range(int(num_pieces)):
    k_list.append(st.sidebar.slider(f'k{p}', min_value=-3, max_value=3, value=int(k_init[p])))

d = st.sidebar.slider('Shift parameter, d', min_value=-10, max_value=10, value=0)
xi_list = np.arange(len(k_list)-1)


st.subheader('Phase portrait of the map')

fig_map, ax_map = plt.subplots(figsize=(5, 5))
ax_map.set_xlabel(r'$q$')
ax_map.set_ylabel(r'$p$')
ax_map = plot_orbits(ax_map, k_list, xi_list, d=d, Tmax=int(Tmax))
ax_map.set_aspect('equal')
st.pyplot(fig_map)

st.sidebar.subheader('Force function')
x = np.linspace(min(xi_list)-2, max(xi_list)+2, 1000)

# plot force function
# fig_f, ax_f = plt.subplots()
# ax_f.set_xlabel(r'$q$')
# ax_f.set_ylabel(r'$f(q)$')
f = []
for xi in x:
    f.append(force(xi, k_list, xi_list, d))

# ax_f.plot(x, f, color='k')
df = pd.DataFrame({'q': x, 'f(q)': f})
fig_f = px.line(df, x='q', y='f(q)', width=300, height=300)
st.sidebar.plotly_chart(fig_f, use_container_width=True)

st.markdown("#### Preprint: [arXiv: 2201.13133](https://arxiv.org/abs/2201.13133)")
st.markdown("""#### Contributors:

                * Yaroslav Kharkov (Univeristy of Maryland)
                * Timothy Zolkin (Fermilab)
                """)
