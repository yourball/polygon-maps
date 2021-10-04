import streamlit as st
import pandas as pd
import numpy as np
# import plotly.figure_factory as ff
# import plotly.express as px
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
    for x0 in x0_list:
        y0 = x0*1.01
        traj_data = orbit(x0, y0, k_list, xi_list, d, Tmax=Tmax)

        ax.scatter(traj_data[:, 0], traj_data[:, 1], s=1,
                   color=[abs(x0)/max_x0, .1*abs(x0)/max_x0, 1-abs(x0)/max_x0])
    return ax

st.title('Integrable polygonal mappings of the plane')

k_list = []

num_pieces = st.number_input('Enter a number of piecewise regions',
                             min_value=2, max_value=10, value=3, step=1)
st.text(r'Slopes of the piesewise function')
for p in range(num_pieces):
    k_list.append(st.slider(f'k{p}', min_value=-3, max_value=3, value=0))

d = st.slider('Shift parameter, d', min_value=-10, max_value=10, value=0)
xi_list = np.arange(len(k_list)-1)

fig_map, ax_map = plt.subplots()
ax_map.set_xlabel('q')
ax_map.set_ylabel('p')
ax_map = plot_orbits(ax_map, k_list, xi_list, d=d, Tmax=1000)
st.pyplot(fig_map)

st.write('Force function')
# plot force function
fig_f, ax_f = plt.subplots()
x = np.linspace(min(xi_list)-2, max(xi_list)+2, 1000)
ax_f.set_xlabel('q')
ax_f.set_ylabel('f(q)')
f = []
for xi in x:
    f.append(force(xi, k_list, xi_list, d))
ax_f.plot(x, f)

st.pyplot(fig_f)
