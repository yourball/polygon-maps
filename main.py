import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
from matplotlib.cm import get_cmap

epsV = 1e-4
epsT = 1e-10
Tmax = 2000
Vmax = 200
Rmax = 200
x0_edge = 20 + np.pi * 1e-2


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


def vertex_counter(data, qstar):
    x_c = (np.max(data[:, 0]) + np.min(data[:, 0])) / 2.0
    y_c = (np.max(data[:, 1]) + np.min(data[:, 1])) / 2.0

    # x_c, y_c = qstar, qstar
    angles = np.angle(data[:, 0] - x_c + 1j * (data[:, 1] - y_c))
    indx = np.argsort(angles)
    x_sorted, y_sorted = data[indx, 0], data[indx, 1]

    delta_x = x_sorted[1:] - x_sorted[:-1]
    delta_y = y_sorted[1:] - y_sorted[:-1]

    dot_prods = delta_x[1:] * delta_x[:-1] + delta_y[1:] * delta_y[:-1]
    alphas = dot_prods / (
        1e-16
        + np.sqrt(delta_x[1:] ** 2 + delta_y[1:] ** 2)
        * np.sqrt(delta_x[:-1] ** 2 + delta_y[:-1] ** 2)
    )
    V = len(alphas[np.abs(alphas - 1) > epsV])
    return V


# @njit(parallel=True)
def orbit_classif(
    qstar, x0, y0, k_list, xi_list, ai_list, bi_list, d_shift=0, Tmax=Tmax, Rmax=Rmax
):
    T, V, stab = 0, 0, True
    x, y = x0, y0
    traj_list = [[x0, y0]]
    for iter in range(Tmax):
        x, y = y, -x + force(y, k_list, xi_list, d_shift)
        # traj_data = np.vstack([traj_data, [x, y]])
        traj_list += [[x, y]]
        T += 1
        if np.abs(x - x0) < epsT and np.abs(y - y0) < epsT:
            # period is detected
            stab = True
            break
        if (x - x0) ** 2 + (y - y0) ** 2 > Rmax ** 2:
            # trajectory goes outside of the circle with radius Rmax
            stab = False
            break
    traj_data = np.vstack(traj_list)
    if stab:
        V = vertex_counter(traj_data, qstar)
    return stab, T, V, traj_data


def domain_analyzer(
    orbits_data, report_dict, k_list, xi_list, d_shift, exit_flag, d_shift_type
):
    str_k = ",".join([str(round(elem, 2)) for elem in k_list])
    str_d = str(d_shift)
    if exit_flag:
        stable = False
        integr = False
    else:
        stab_i, T_i, V_i = orbits_data[:, 0], orbits_data[:, 1], orbits_data[:, 2]
        indx_stable = np.where(stab_i == 1)

        V_i_stable = V_i[indx_stable]
        T_i_stable = T_i[indx_stable]
        stable = (stab_i).all()
        report_dict["stable"][str_k] = stable
        if not stable:
            integr = False
        else:
            linear = (np.abs(T_i_stable[1:] - T_i_stable[0:-1]) < epsT).all() and (
                T_i_stable < Tmax
            ).all()
            if linear:
                integr = True
            else:
                integr = (V_i_stable < Vmax).all()

    report_dict["integr"][str_k][str_d] = integr

    report_dict['xi_list'] = xi_list
    # report_dict['stable'][str_k][str_d] = stable
    return report_dict


# @njit
def map_analyser(
    num_pieces=2,
    ki_max_range=3,
    dki=1,
    d_shift_min=-2,
    d_shift_max=2,
    vis=False,
    d_shift_type="discrete",
):
    xi_list = np.arange(num_pieces - 1)
    report_dict = {}

    report_dict["stable"] = {}
    report_dict["integr"] = {}

    _ki = np.arange(-ki_max_range, ki_max_range + dki, dki)

    _ki_list = [_ki for i in range(num_pieces)]
    _ki_kron = list(itertools.product(*_ki_list))

    n_d = d_shift_max - d_shift_min
    if d_shift_type == "discrete":
        _d_shift = np.arange(d_shift_min, d_shift_max + 1)
    elif d_shift_type == "continous":
        d_shift_range = d_shift_max - d_shift_min
        _d_shift = d_shift_range * (0.5 - np.random.rand(n_d))

    counter = 0
    for i, k_list in enumerate(_ki_kron):

        str_k = ",".join([str(round(elem, 2)) for elem in k_list])
        report_dict["integr"][str_k] = {}

        # Check if can be reduced to smaller number of segments
        # If yes, assign None
        k_vec = np.array(k_list)
        if np.any(k_vec[1:] - k_vec[:-1] == 0):
            report_dict["integr"][str_k] = np.nan
            continue

        for d_shift in _d_shift:
            orbits_data = np.array([])
            exit_flag = False
            qstar = find_qstar(force, k_list, xi_list, d_shift)

            print('k: ', k_list, ' d: ', d_shift, 'q*: ', qstar)
            if qstar is np.nan:
                exit_flag = True
            #
            # if (qstar <= xi_list[0]):
            #     x0_min, x0_max = qstar-3, xi_list[-1]+3
            # elif (qstar >= xi_list[-1]):
            #     x0_min, x0_max = xi_list[-1]-3, qstar+3
            # else:
            #     x0_min, x0_max = xi_list[0] - 3, xi_list[-1] + 3
            x0_min, x0_max = -x0_edge - 1 * abs(d_shift), x0_edge + 1 * abs(d_shift)

            print('x0_min: ', x0_min, ' x0_max: ', x0_max)
            x0_vec = np.linspace(x0_min, x0_max, 100)
            stab = True
            for x0 in x0_vec:
                if not stab:
                    exit_flag = True
                    break
                y0 = x0*1.0243198 + 1e-2*np.pi
                stab, T, V, traj_data = orbit_classif(
                    qstar, x0, y0, k_list, xi_list, ai_list, bi_list, d_shift=d_shift
                )
                orbits_data_i = np.array([stab, T, V])

                if orbits_data.size == 0:
                    orbits_data = orbits_data_i
                else:
                    orbits_data = np.vstack([orbits_data, orbits_data_i])
                if exit_flag:
                    break
                if vis:
                    counter += 1
                    plot_info = {}
                    plot_info["counter"] = counter
                    plot_info["k_list"] = k_list.tolist()
                    plot_info["x_i_list"] = xi_list.tolist()
                    plot_info["d_shift"] = d_shift
                    plot_info["T"] = T
                    plot_info["V"] = V
                    plot_info["x0"] = x0
                    plot_info["y0"] = y0
                    plot_trajs(traj_data, plot_info)
            report_dict = domain_analyzer(
                orbits_data,
                report_dict,
                k_list,
                xi_list,
                d_shift,
                exit_flag,
                d_shift_type=d_shift_type,
            )

    return report_dict


def plot_trajs(traj_data, plot_info=None):
    fig, ax = plt.subplots(1, 1)
    name = "Paired"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    ax.set_prop_cycle(color=colors)
    k_i = ",".join([str(round(elem, 2)) for elem in plot_info["k_list"]])
    x_i = ",".join([str(round(elem, 2)) for elem in plot_info["x_i_list"]])
    x0, y0 = plot_info["x0"], plot_info["y0"]
    info = "T:{:}, V:{:}, force=>k_i:({:s}),x_i:({:s})".format(
        plot_info["T"], plot_info["V"], k_i, x_i
    )
    info_add = " (x0:{0:.2f}, y0:{0:.2f})".format(x0, y0)
    info += info_add
    ax.scatter(traj_data[:, 0], traj_data[:, 1], label=info)
    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$y$", fontsize=16)
    ax.set_title("Polygon map", fontsize=16)
    plt.legend(loc=1)
    plt.savefig("figs/polyg_{:}.png".format(plot_info["counter"]))

    plt.draw()
    plt.pause(0.01)
    plt.close()


def report_plot(report_dict, ki_max_range):
    nk_i = 2 * ki_max_range + 1
    for report_type in ["integr", "linear", "stable"]:
        ki_ = [*report_dict["integr"]]
        ki_table = np.array(ki_).reshape(nk_i, nk_i)
        table = np.array([])
        for ki in ki_:
            table = np.append(table, report_dict[report_type][ki])

        table = table.reshape(nk_i, nk_i)

        plt.imshow(table)
        plt.colorbar()
        plt.title(report_type)
        plt.xticks(None)

        for i in range(nk_i):
            for j in range(nk_i):
                text = plt.text(
                    j, i, ki_table[i, j], ha="center", va="center", color="w"
                )
        plt.savefig("./report_{:}.png".format(report_type))

        plt.draw()
        plt.pause(10)
        plt.close()


def report_plot3_d(report_dict, ki_max_range, d_shift_type="discrete"):
    nk_i = 2 * ki_max_range + 1

    table_all_d = np.array([])
    table_d_vals = np.array([])
    report_type = "integr"

    fig, axs = plt.subplots(1, nk_i, sharex=True, sharey=True)

    _ki = np.arange(-ki_max_range, ki_max_range + 1)
    _ki_list_out = [_ki for i in range(2)]
    _ki_kron_out = list(itertools.product(*_ki_list_out))
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color="black")
    for i_mid, k_mid in enumerate(_ki):
        table_all_d = np.array([])
        table_d_vals = np.array([])
        some_d = np.array([])
        ki_table = []

        for ki in _ki_kron_out:

            k_vec = np.insert(ki, 1, k_mid)
            str_k = ",".join([str(elem) for elem in k_vec])
            # Check if can be reduced to smaller number of segments
            has_repeated_ki = np.any(k_vec[1:] - k_vec[:-1] == 0)
            if has_repeated_ki:
                integr_all_d = np.nan
            else:
                integr_d = np.array(list(report_dict[report_type][str_k].values()))
                integr_all_d = integr_d.all()

            table_all_d = np.append(table_all_d, integr_all_d)
            ki_table.append(str_k)

            if not has_repeated_ki:
                indx = np.where(integr_d)[0]
                if len(indx) == 0:
                    d_str = ""
                else:
                    d_list = np.array([*report_dict[report_type][str_k]])
                    d_list = d_list[indx]
                    d_str = ";".join([elem for elem in d_list])
                if np.any(k_vec[1:] - k_vec[:-1] == 0):
                    d_str = ""
                table_d_vals = np.append(table_d_vals, d_str)
                some_d = np.append(some_d, len(d_str) > 0)
                some_d = np.array(some_d, dtype=int)
            else:
                table_d_vals = np.append(table_d_vals, np.nan)
                some_d = np.append(some_d, np.nan)

        table_all_d = table_all_d.reshape(nk_i, nk_i)
        table_d_vals = table_d_vals.reshape(nk_i, nk_i)
        ki_table = np.array(ki_table).reshape(nk_i, nk_i)
        some_d = some_d.reshape(nk_i, nk_i)

        axs[i_mid].set_title("k1=" + str(k_mid), fontsize=8)
        axs[i_mid].set_adjustable("box")
        axs[i_mid].get_xaxis().set_visible(False)
        axs[i_mid].get_yaxis().set_visible(False)

        table_sum = np.array(table_all_d, dtype=float)

        axs[i_mid].matshow(table_sum)
        for i in range(nk_i):
            for j in range(nk_i):
                if (
                    d_shift_type == "discrete"
                    and len(table_d_vals[i, j]) > 0
                    and not table_all_d[i, j]
                ):
                    text = axs[i_mid].text(
                        j,
                        i,
                        table_d_vals[i, j],
                        ha="center",
                        va="bottom",
                        color="w",
                        fontsize=2,
                    )
    plt.savefig("./report3p_{:}_{:}.eps".format(d_shift_type, report_type))

    plt.draw()
    plt.pause(5)
    plt.close()


def report_plot4_d(report_dict, ki_max_range, d_shift_type="discrete"):
    nk_i = 2 * ki_max_range + 1

    table_all_d = np.array([])
    table_d_vals = np.array([])
    report_type = "integr"

    fig, axs = plt.subplots(nk_i, nk_i, sharex=True, sharey=True)

    _ki = np.arange(-ki_max_range, ki_max_range + 1)
    _ki_list_out = [_ki for i in range(2)]
    _ki_kron_out = list(itertools.product(*_ki_list_out))
    # plt.setp(axs, xticks=np.arange(nk_i), yticks=np.arange(nk_i), xticklabels=None, yticklabels=None)

    current_cmap = plt.cm.get_cmap()
    current_cmap.set_bad(color="black")
    for i_out1, k_out1 in enumerate(_ki):
        for i_out2, k_out2 in enumerate(_ki):
            table_all_d = np.array([])
            table_d_vals = np.array([])
            some_d = np.array([])
            table_sum = np.array([])
            k_table = []
            for ki in _ki_kron_out:

                k_vec = np.insert(ki, 0, k_out1)
                k_vec = np.append(k_vec, k_out2)

                str_k = ",".join([str(elem) for elem in k_vec])
                k_table.append(str_k)
                # Check if can be reduced to smaller number of segments
                has_repeated_ki = np.any(k_vec[1:] - k_vec[:-1] == 0)
                if has_repeated_ki:
                    integr_all_d = np.nan
                else:
                    integr_d = np.array(list(report_dict[report_type][str_k].values()))
                    integr_all_d = integr_d.all()

                table_all_d = np.append(table_all_d, integr_all_d)
                if not has_repeated_ki:
                    indx = np.where(integr_d)[0]
                    if len(indx) == 0:
                        d_str = ""
                    else:
                        d_list = np.array([*report_dict[report_type][str_k]])
                        d_list = d_list[indx]
                        d_str = ";".join([elem for elem in d_list])
                    if np.any(k_vec[1:] - k_vec[:-1] == 0):
                        d_str = ""
                    table_d_vals = np.append(table_d_vals, d_str)
                    some_d = np.append(some_d, len(d_str) > 0)
                    some_d = np.array(some_d, dtype=int)
                else:
                    table_d_vals = np.append(table_d_vals, np.nan)
                    some_d = np.append(some_d, np.nan)

            table_all_d = table_all_d.reshape(nk_i, nk_i)
            table_d_vals = table_d_vals.reshape(nk_i, nk_i)
            some_d = some_d.reshape(nk_i, nk_i)
            k_table = np.array(k_table).reshape(nk_i, nk_i)
            # axs[i_out2, i_out1].set_title(
            #     "k0=" + str(k_out1) + " k3=" + str(k_out2), fontsize=1
            # )
            axs[i_out2, i_out1].set_adjustable("box")
            table_sum = np.array(table_all_d, dtype=float)
            axs[i_out2, i_out1].matshow(0*table_sum)
            axs[i_out2, i_out1].get_xaxis().set_visible(False)
            axs[i_out2, i_out1].get_yaxis().set_visible(False)
            for i in range(nk_i):
                for j in range(nk_i):
                    text = axs[i_out2, i_out1].text(j, i, k_table[i, j], ha="center", va="top", color="w", fontsize=.5)

                    # if (
                    #     d_shift_type == "discrete"
                    #     and len(table_d_vals[i, j]) > 0
                    #     and not table_all_d[i, j]
                    # ):

                        # text = axs[i_out2, i_out1].text(
                        #     j,
                        #     i,
                        #     table_d_vals[i, j],
                        #     ha="center",
                        #     va="bottom",
                        #     color="w",
                        #     fontsize=2,
                        # )
    plt.savefig("./report4p_{:}_{:}.pdf".format(d_shift_type, report_type))
    plt.savefig("./report4p_{:}_{:}.png".format(d_shift_type, report_type))
    plt.draw()
    plt.pause(5)
    plt.close()

def map_analyser_custom(
    ki_list,
    ri_list,
    d_shift_min=-2,
    d_shift_max=2,
    d_shift_type="discrete",
):
    num_pieces = np.array(ki_list).shape[1]
    report_dict = {}

    report_dict["stable"] = {}
    report_dict["integr"] = {}

    n_d = d_shift_max - d_shift_min
    if d_shift_type == "discrete":
        _d_shift = np.arange(d_shift_min, d_shift_max + 1)
    elif d_shift_type == "continous":
        d_shift_range = d_shift_max - d_shift_min
        _d_shift = d_shift_range * (0.5 - np.random.rand(n_d))

    counter = 0
    for ri in ri_list:
        xi_list = np.array([0, 1, 1 + ri])  # Customize ratio of segments, 4-piece function
        for i, k_list in enumerate(ki_list):

            str_k = ",".join([str(round(elem, 2)) for elem in k_list])
            report_dict["integr"][str_k] = {}

            for d_shift in _d_shift:
                V_list = []
                orbits_data = np.array([])
                exit_flag = False
                qstar = find_qstar(force, k_list, xi_list, d_shift)

                if qstar is np.nan:
                    exit_flag = True

                x0_min, x0_max = -x0_edge - .75 * abs(d_shift), x0_edge + .75 * abs(d_shift)

                # print('x0_min: ', x0_min, ' x0_max: ', x0_max)
                x0_vec = np.linspace(x0_min, x0_max, 100)
                stab = True
                for x0 in x0_vec:
                    if not stab:
                        exit_flag = True
                        break
                    y0 = x0*1.0943198 + 1e-2*np.pi
                    stab, T, V, traj_data = orbit_classif(
                        qstar, x0, y0, k_list, xi_list, ai_list, bi_list, d_shift=d_shift
                    )
                    orbits_data_i = np.array([stab, T, V])
                    V_list.append(V)
                    if orbits_data.size == 0:
                        orbits_data = orbits_data_i
                    else:
                        orbits_data = np.vstack([orbits_data, orbits_data_i])
                    if exit_flag:
                        break
                report_dict = domain_analyzer(
                    orbits_data,
                    report_dict,
                    k_list,
                    xi_list,
                    d_shift,
                    exit_flag,
                    d_shift_type=d_shift_type,
                )
                print('k_i: ', k_list, ' d: ', d_shift, 'ri:', ri, 'Integrable:', report_dict["integr"][str_k][str(d_shift)])
                print('V_list', V_list)
    return report_dict


def find_qstar(force_f, k_list, xi_list, d_shift):
    # Find root of equation f(q^*)/2 = q^*
    q_arr = np.array(xi_list)
    q_arr = np.insert(q_arr, 0, xi_list[0] - 100)
    q_arr = np.append(q_arr, xi_list[-1] + 100)

    ai_list, bi_list = precompute_force(k_list, xi_list)
    f_list = []
    for x in q_arr:
        f = force(x, k_list, xi_list, d_shift, ai_list, bi_list)
        f_list.append(f)
    # Check whether LHS - RHS is positive or negative
    sign_list = np.array(np.array(f_list) / 2 - q_arr >= 0, dtype=int)
    # Find where LHS - RHS changes sign
    idx_vec = np.where(sign_list[1:] - sign_list[:-1] != 0)[0]

    # import pdb; pdb.set_trace()
    # No q* is detected
    if len(idx_vec) == 0:
        return np.nan
    # More then one q* is detected
    if len(idx_vec) > 1:
        return np.nan

    idx = idx_vec[0]
    ki = k_list[idx]
    # check if it is an external segment

    num = (bi_list[idx] - ki*ai_list[idx] + d_shift)
    if ki == 2 and num != 0:
        return np.nan
    if ki == 2 and num == 0:
        return .5*(q_arr[idx] + q_arr[idx+1])
    else:
        qstar = num / (2 - ki)
        lhs_m_rhs = force(qstar, k_list, xi_list, d_shift, ai_list, bi_list) - 2*qstar
        # if abs(lhs_m_rhs) > 1e-15:
        #     print('f(qstar) - 2*qstar: ', lhs_m_rhs)
    return qstar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters")
    parser.add_argument("--vis", "-v", action="store_true")
    parser.add_argument(
        "--d_shift_type",
        "-d",
        type=str,
        choices=["discrete", "continous"],
        default="discrete",
    )
    parser.add_argument("--load", "-l", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    d_shift_type = args.d_shift_type

    # Calculate report and save dict
    ki_max_range = 2
    d_shift_min = -3
    d_shift_max = 5

    num_pieces = 3
    fname = "report_dict{:}p_{:}.npy".format(num_pieces, d_shift_type)

    # commented out code below
    # if not args.load:
    #     report_dict = map_analyser(
    #         num_pieces=num_pieces,
    #         ki_max_range=ki_max_range,
    #         dki=1,
    #         d_shift_min=d_shift_min,
    #         d_shift_max=d_shift_max,
    #         vis=args.vis,
    #         d_shift_type=d_shift_type,
    #     )
    #     # Save
    #     np.save(fname, report_dict)
    # else:
    #     # Load
    #     report_dict = np.load(fname, allow_pickle="TRUE").item()
    # if args.verbose:
    #     print(report_dict)

    # report_plot3_d(report_dict, ki_max_range=ki_max_range, d_shift_type=d_shift_type)
    # report_plot4_d(report_dict, ki_max_range=ki_max_range, d_shift_type=d_shift_type)

    ki_list = [
               # [-1, -2, -1, -2],
               # [0, -1, 0, -1],
               # [1, 0, 1, 0],
               [1, 2, 1, 0],
               # [1, 0, -1, 0],
               # [-2, -1, -2, -1],
               # [-1, 0, -1, 0],
               # [0, 1, 0, 1],
               # [0, 1, 2, 1],
               # [0, -1, 0, 1]
               ]
    ri_list = [9] #np.arange(1, 10)
    map_analyser_custom(
        ki_list,
        ri_list,
        d_shift_min=-20,
        d_shift_max=20,
        d_shift_type="discrete")
