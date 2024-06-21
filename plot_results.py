from matplotlib.pyplot import *
from numpy import loadtxt, array, zeros
import os


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])  #,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def plot_result(U, t_diag, V1, GA, kBT, plot_s_id, plot_for_poster):
    if plot_for_poster:
        U1 = 4.0
        U2 = 8.0
        td1 = 0.0
        td2 = -0.4
        folder_name_GA_U1_td1 = 'results_with_GA' + '_U_' + str(U1) + '_t_diag_' + str(td1) + '_V1_' + str(V1)
        folder_name_GA_U1_td1_zoom = 'results_with_GA' + '_U_' + str(U1) + '_t_diag_' + str(td1) + '_V1_' + str(V1) + '_zoom'
        folder_name_GA_U2_td1 = 'results_with_GA' + '_U_' + str(U2) + '_t_diag_' + str(td1) + '_V1_' + str(V1)
        folder_name_GA_U1_td2 = 'results_with_GA' + '_U_' + str(U1) + '_t_diag_' + str(td2) + '_V1_' + str(V1)
        folder_name_GA_U2_td2 = 'results_with_GA' + '_U_' + str(U2) + '_t_diag_' + str(td2) + '_V1_' + str(V1)

        path_GA_U1_td1 = './' + folder_name_GA_U1_td1 + '/'
        path_GA_U1_td1_zoom = './' + folder_name_GA_U1_td1_zoom + '/'
        path_GA_U2_td1 = './' + folder_name_GA_U2_td1 + '/'
        path_GA_U1_td2 = './' + folder_name_GA_U1_td2 + '/'
        path_GA_U2_td2 = './' + folder_name_GA_U2_td2 + '/'

        str_part_GA = '_with_GA'

        folder_name_HF_U1_td1 = 'results_no_GA' + '_U_' + str(U1) + '_t_diag_' + str(td1) + '_V1_' + str(V1)
        folder_name_HF_U1_td1_zoom = 'results_no_GA' + '_U_' + str(U1) + '_t_diag_' + str(td1) + '_V1_' + str(V1) + '_zoom'
        folder_name_HF_U2_td1 = 'results_no_GA' + '_U_' + str(U2) + '_t_diag_' + str(td1) + '_V1_' + str(V1)
        folder_name_HF_U1_td2 = 'results_no_GA' + '_U_' + str(U1) + '_t_diag_' + str(td2) + '_V1_' + str(V1)
        folder_name_HF_U2_td2 = 'results_no_GA' + '_U_' + str(U2) + '_t_diag_' + str(td2) + '_V1_' + str(V1)

        path_HF_U1_td1 = './' + folder_name_HF_U1_td1 + '/'
        path_HF_U1_td1_zoom = './' + folder_name_HF_U1_td1_zoom + '/'
        path_HF_U2_td1 = './' + folder_name_HF_U2_td1 + '/'
        path_HF_U1_td2 = './' + folder_name_HF_U1_td2 + '/'
        path_HF_U2_td2 = './' + folder_name_HF_U2_td2 + '/'

        str_part_HF = '_no_GA'


        # GA data
        Delta_s_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_Delta_s' + str_part_GA))
        Delta_d_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_Delta_d' + str_part_GA))
        Delta_s_s_id_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_Delta_s_s_id' + str_part_GA))
        Delta_d_s_id_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_Delta_d_s_id' + str_part_GA))

        # for the zoom
        Delta_s_GA_U1_td1_zoom = loadtxt(os.path.join(path_GA_U1_td1_zoom, 'res_Delta_s' + str_part_GA))
        Delta_d_GA_U1_td1_zoom = loadtxt(os.path.join(path_GA_U1_td1_zoom, 'res_Delta_d' + str_part_GA))
        Delta_s_s_id_GA_U1_td1_zoom = loadtxt(os.path.join(path_GA_U1_td1_zoom, 'res_Delta_s_s_id' + str_part_GA))
        Delta_d_s_id_GA_U1_td1_zoom = loadtxt(os.path.join(path_GA_U1_td1_zoom, 'res_Delta_d_s_id' + str_part_GA))

        Delta_s_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_Delta_s' + str_part_GA))
        Delta_d_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_Delta_d' + str_part_GA))
        Delta_s_s_id_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_Delta_s_s_id' + str_part_GA))
        Delta_d_s_id_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_Delta_d_s_id' + str_part_GA))

        Delta_s_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_Delta_s' + str_part_GA))
        Delta_d_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_Delta_d' + str_part_GA))
        Delta_s_s_id_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_Delta_s_s_id' + str_part_GA))
        Delta_d_s_id_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_Delta_d_s_id' + str_part_GA))

        Delta_s_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_Delta_s' + str_part_GA))
        Delta_d_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_Delta_d' + str_part_GA))
        Delta_s_s_id_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_Delta_s_s_id' + str_part_GA))
        Delta_d_s_id_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_Delta_d_s_id' + str_part_GA))



        # HF data
        Delta_s_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_Delta_s' + str_part_HF))
        Delta_d_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_Delta_d' + str_part_HF))
        Delta_s_id_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_Delta_s_s_id' + str_part_HF))
        Delta_d_s_id_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_Delta_d_s_id' + str_part_HF))

        # for the zoom
        Delta_s_HF_U1_td1_zoom = loadtxt(os.path.join(path_HF_U1_td1_zoom, 'res_Delta_s' + str_part_HF))
        Delta_d_HF_U1_td1_zoom = loadtxt(os.path.join(path_HF_U1_td1_zoom, 'res_Delta_d' + str_part_HF))
        Delta_s_s_id_HF_U1_td1_zoom = loadtxt(os.path.join(path_HF_U1_td1_zoom, 'res_Delta_s_s_id' + str_part_HF))
        Delta_d_s_id_HF_U1_td1_zoom = loadtxt(os.path.join(path_HF_U1_td1_zoom, 'res_Delta_d_s_id' + str_part_HF))

        Delta_s_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_Delta_s' + str_part_HF))
        Delta_d_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_Delta_d' + str_part_HF))
        Delta_s_id_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_Delta_s_s_id' + str_part_HF))
        Delta_d_s_id_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_Delta_d_s_id' + str_part_HF))

        Delta_s_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_Delta_s' + str_part_HF))
        Delta_d_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_Delta_d' + str_part_HF))
        Delta_s_id_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_Delta_s_s_id' + str_part_HF))
        Delta_d_s_id_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_Delta_d_s_id' + str_part_HF))

        Delta_s_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_Delta_s' + str_part_HF))
        Delta_d_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_Delta_d' + str_part_HF))
        Delta_s_id_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_Delta_s_s_id' + str_part_HF))
        Delta_d_s_id_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_Delta_d_s_id' + str_part_HF))

        cidciu_s_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_cidciu_s' + str_part_GA))
        cidciu_d_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_cidciu_d' + str_part_GA))
        cidciu_s_id_GA_U1_td1 = loadtxt(os.path.join(path_GA_U1_td1, 'res_cidciu_s_id' + str_part_GA))

        cidciu_s_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_cidciu_s' + str_part_GA))
        cidciu_d_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_cidciu_d' + str_part_GA))
        cidciu_s_id_GA_U2_td1 = loadtxt(os.path.join(path_GA_U2_td1, 'res_cidciu_s_id' + str_part_GA))

        cidciu_s_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_cidciu_s' + str_part_GA))
        cidciu_d_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_cidciu_d' + str_part_GA))
        cidciu_s_id_GA_U1_td2 = loadtxt(os.path.join(path_GA_U1_td2, 'res_cidciu_s_id' + str_part_GA))

        cidciu_s_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_cidciu_s' + str_part_GA))
        cidciu_d_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_cidciu_d' + str_part_GA))
        cidciu_s_id_GA_U2_td2 = loadtxt(os.path.join(path_GA_U2_td2, 'res_cidciu_s_id' + str_part_GA))

        cidciu_s_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_cidciu_s' + str_part_HF))
        cidciu_d_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_cidciu_d' + str_part_HF))
        cidciu_s_id_HF_U1_td1 = loadtxt(os.path.join(path_HF_U1_td1, 'res_cidciu_s_id' + str_part_HF))

        cidciu_s_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_cidciu_s' + str_part_HF))
        cidciu_d_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_cidciu_d' + str_part_HF))
        cidciu_s_id_HF_U2_td1 = loadtxt(os.path.join(path_HF_U2_td1, 'res_cidciu_s_id' + str_part_HF))

        cidciu_s_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_cidciu_s' + str_part_HF))
        cidciu_d_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_cidciu_d' + str_part_HF))
        cidciu_s_id_HF_U1_td2 = loadtxt(os.path.join(path_HF_U1_td2, 'res_cidciu_s_id' + str_part_HF))

        cidciu_s_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_cidciu_s' + str_part_HF))
        cidciu_d_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_cidciu_d' + str_part_HF))
        cidciu_s_id_HF_U2_td2 = loadtxt(os.path.join(path_HF_U2_td2, 'res_cidciu_s_id' + str_part_HF))

        density_array = loadtxt(os.path.join(path_GA_U1_td1, 'res_density_s' + str_part_GA))

        density_array_zoom = loadtxt(os.path.join(path_GA_U1_td1_zoom, 'res_density_s' + str_part_GA))


        # plots for t_diag = 0.0
        lw = 3
        ls_GA_s = '-b'
        ls_GA_d = '-r'
        ls_GA_s_id = '-g'
        ls_HF_s = '--b'
        ls_HF_d = '--r'
        ls_HF_s_id = '--g'
        fs = 20
        opaci_HF = 0.6

        fig, ax = subplots(2, 2, figsize = (10, 10))
        fig.tight_layout(pad=4.0)


        # td1 plots
        ax[0,0].set_title(r'$U/t$ = ' + str(int(U1)) + r', $t^\prime/t$ = ' + str(td1), fontsize = fs)
        ax[0,0].plot(density_array, array(Delta_s_GA_U1_td1)/V1, ls_GA_s, linewidth = lw, label = 'GA - s')
        ax[0,0].plot(density_array, array(Delta_d_GA_U1_td1)/V1, ls_GA_d, linewidth = lw, label = 'GA - d')
        ax[0,0].plot(density_array, array(abs(Delta_s_s_id_GA_U1_td1 / V1 + 1j* Delta_d_s_id_GA_U1_td1 / V1)), ls_GA_s_id, linewidth = lw, label = 'GA - s+id')
        ax[0,0].plot(density_array, array(Delta_s_HF_U1_td1)/V1, ls_HF_s, linewidth = lw, alpha = opaci_HF, label = 'HF - s')
        ax[0,0].plot(density_array, array(Delta_d_HF_U1_td1)/V1, ls_HF_d, linewidth = lw, alpha = opaci_HF, label = 'HF - d')
        ax[0,0].plot(density_array, array(abs(Delta_s_id_HF_U1_td1 / V1 + 1j* Delta_d_s_id_HF_U1_td1 / V1)), ls_HF_s_id, linewidth = lw, alpha = opaci_HF, label = 'HF - s+id')
        ax[0,0].set_ylabel(r'$\Delta_s$, $\Delta_d$, $\Delta_{s+id}$', fontsize = fs)
        ax[0,0].grid()




        # arrow to the zoom plot
        ax[0, 0].arrow(0.38, 0.01, 0.275, 0.016, head_width=0.01, head_length=0.05, fc='k', ec='k', lw=1.5, ls='-',
                       length_includes_head=True)
        ax[0,0].set_axisbelow(True)
        # this is a zoom inside the first plot
        rect = [0.6, 0.2, 0.38, 0.38]
        x_left = 10
        x_right = 40
        lw_subsub_plot = 1.5
        opaci_for_subplot = 0.5
        ls_GA_s_s_id = '-m'
        ls_GA_d_s_id = '-c'
        ls_HF_s_s_id = '--m'
        ls_HF_d_s_id = '--c'
        ax1 = add_subplot_axes(ax[0,0], rect)


        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_s_s_id_GA_U1_td1_zoom)[x_left: x_right] / V1, ls_GA_s_s_id,
                 linewidth=lw_subsub_plot, alpha = opaci_for_subplot)
        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_d_s_id_GA_U1_td1_zoom)[x_left: x_right] / V1, ls_GA_d_s_id,
                 linewidth=lw_subsub_plot, alpha = opaci_for_subplot)
        ax1.legend([r'$\Delta^{\prime}_s$', r'$\Delta^{\prime}_d$'], fontsize=fs - 12)

        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_s_s_id_HF_U1_td1_zoom)[x_left: x_right] / V1,
                 ls_HF_s_s_id,
                 linewidth=lw_subsub_plot, alpha=opaci_for_subplot)
        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_d_s_id_HF_U1_td1_zoom)[x_left: x_right] / V1,
                 ls_HF_d_s_id,
                 linewidth=lw_subsub_plot, alpha=opaci_for_subplot)

        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_s_GA_U1_td1_zoom)[x_left: x_right] / V1, ls_GA_s,
                 linewidth=lw_subsub_plot)
        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_d_GA_U1_td1_zoom)[x_left: x_right] / V1, ls_GA_d,
                 linewidth=lw_subsub_plot)

        ax1.plot(density_array_zoom[x_left: x_right], array(abs(Delta_s_s_id_GA_U1_td1_zoom / V1 + 1j* Delta_d_s_id_GA_U1_td1_zoom / V1))[x_left: x_right], ls_GA_s_id, linewidth = lw_subsub_plot)

        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_s_HF_U1_td1_zoom)[x_left: x_right]/V1, ls_HF_s, linewidth = lw_subsub_plot, alpha = opaci_HF)
        ax1.plot(density_array_zoom[x_left: x_right], array(Delta_d_HF_U1_td1_zoom)[x_left: x_right]/V1, ls_HF_d, linewidth = lw_subsub_plot, alpha = opaci_HF)



        ax1.plot(density_array_zoom[x_left: x_right], array(abs(Delta_s_s_id_HF_U1_td1_zoom / V1 + 1j* Delta_d_s_id_HF_U1_td1_zoom / V1))[x_left: x_right], ls_HF_s_id, linewidth = lw_subsub_plot, alpha = opaci_HF)


        ax[0,1].set_title(r'$U/t$ = ' + str(int(U2)) + r', $t^\prime/t$ = ' + str(td1), fontsize = fs)
        ax[0,1].plot(density_array, array(Delta_s_GA_U2_td1)/V1, ls_GA_s, linewidth = lw, label = 'GA - s')
        ax[0,1].plot(density_array, array(Delta_d_GA_U2_td1)/V1, ls_GA_d, linewidth = lw, label = 'GA - d')
        ax[0,1].plot(density_array, array(abs(Delta_s_s_id_GA_U2_td1 / V1 + 1j* Delta_d_s_id_GA_U2_td1 / V1)), ls_GA_s_id, linewidth = lw, label = 'GA - s+id')
        ax[0,1].plot(density_array, array(Delta_s_HF_U2_td1)/V1, ls_HF_s, linewidth = lw, alpha = opaci_HF, label = 'HF - s')
        ax[0,1].plot(density_array, array(Delta_d_HF_U2_td1)/V1, ls_HF_d, linewidth = lw, alpha = opaci_HF, label = 'HF - d')
        ax[0,1].plot(density_array, array(abs(Delta_s_id_HF_U2_td1 / V1 + 1j* Delta_d_s_id_HF_U2_td1 / V1)), ls_HF_s_id, linewidth = lw, alpha = opaci_HF, label = 'HF - s+id')
        ax[0,1].grid()

        ax[1,0].plot(density_array, cidciu_s_GA_U1_td1, ls_GA_s, linewidth = lw, label = 'GA - s')
        ax[1,0].plot(density_array, cidciu_d_GA_U1_td1, ls_GA_d, linewidth = lw, label = 'GA - d')
        ax[1,0].plot(density_array, cidciu_s_id_GA_U1_td1, ls_GA_s_id, linewidth = lw, label = 'GA - s+id')
        ax[1,0].plot(density_array, cidciu_s_HF_U1_td1, ls_HF_s, linewidth = lw, alpha = opaci_HF, label = 'HF - s')
        ax[1,0].plot(density_array, cidciu_d_HF_U1_td1, ls_HF_d, linewidth = lw, alpha = opaci_HF, label = 'HF - d')
        ax[1,0].plot(density_array, cidciu_s_id_HF_U1_td1, ls_HF_s_id, linewidth = lw, alpha = opaci_HF, label = 'HF - s+id')
        ax[1,0].set_ylabel(r'$\langle c_{i\downarrow} c_{i{\uparrow}} \rangle$', fontsize = fs)
        ax[1, 0].set_xlabel(r'electron density $n$', fontsize=fs)
        ax[1,0].grid()

        ax[1, 1].plot(density_array, cidciu_s_HF_U2_td1, ls_HF_s, linewidth=lw, alpha=opaci_HF, label='HF - s')
        ax[1,1].plot(density_array, cidciu_s_GA_U2_td1, ls_GA_s, linewidth = lw, label = 'GA - s')
        ax[1, 1].plot(density_array, cidciu_d_HF_U2_td1, ls_HF_d, linewidth=lw, alpha=opaci_HF, label='HF - d')
        ax[1,1].plot(density_array, cidciu_d_GA_U2_td1, ls_GA_d, linewidth = lw, label = 'GA - d')
        ax[1, 1].plot(density_array, cidciu_s_id_HF_U2_td1, ls_HF_s_id, linewidth=lw, alpha=opaci_HF, label='HF - s+id')
        ax[1,1].plot(density_array, cidciu_s_id_GA_U2_td1, ls_GA_s_id, linewidth = lw, label = 'GA - s+id')
        ax[1, 1].legend(fontsize = fs)
        ax[1,1].set_xlabel(r'electron density $n$', fontsize = fs)
        ax[1,1].set_axisbelow(True)
        ax[1,1].grid()

        savefig('poster_plots_td1.png')

        show()






        # plots for t_diag = -0.4
        lw = 3
        ls_GA_s = '-b'
        ls_GA_d = '-r'
        ls_GA_s_id = '-g'
        ls_HF_s = '--b'
        ls_HF_d = '--r'
        ls_HF_s_id = '--g'
        fs = 20
        opaci_HF = 0.6

        plot_also_cidciu = False
        if plot_also_cidciu:
            fig, ax = subplots(2, 2, figsize=(10, 10))
            indices1 = [0, 0]
            indices2 = [0, 1]
        else:
            fig, ax = subplots(1, 2, figsize=(10, 5))
            indices1 = [0]
            indices2 = [1]
        fig.tight_layout(pad=4.0)

        # td2 plots
        density_array = loadtxt(os.path.join(path_GA_U1_td2, 'res_density_s' + str_part_GA))
        ax[*indices1].set_title(r'$U/t$ = ' + str(int(U1)) + r', $t^\prime/t$ = ' + str(td2), fontsize=fs)
        ax[*indices1].plot(density_array, array(Delta_s_GA_U1_td2) / V1, ls_GA_s, linewidth=lw)
        ax[*indices1].plot(density_array, array(Delta_d_GA_U1_td2) / V1, ls_GA_d, linewidth=lw)
        ax[*indices1].plot(density_array, array(abs(Delta_s_s_id_GA_U1_td2 / V1 + 1j * Delta_d_s_id_GA_U1_td2 / V1)),
                      ls_GA_s_id, linewidth=lw)
        ax[*indices1].plot(density_array, array(Delta_s_HF_U1_td2) / V1, ls_HF_s, linewidth=lw, alpha=opaci_HF)
        ax[*indices1].plot(density_array, array(Delta_d_HF_U1_td2) / V1, ls_HF_d, linewidth=lw, alpha=opaci_HF)
        ax[*indices1].plot(density_array, array(abs(Delta_s_id_HF_U1_td2 / V1 + 1j * Delta_d_s_id_HF_U1_td2 / V1)),
                      ls_HF_s_id, linewidth=lw, alpha=opaci_HF)
        ax[*indices1].set_ylabel(r'$\Delta_s$, $\Delta_d$, $\Delta_{s+id}$', fontsize=fs)
        ax[*indices1].grid()

        ax[*indices2].set_title(r'$U/t$ = ' + str(int(U2)) + r', $t^\prime/t$ = ' + str(td2), fontsize=fs)
        ax[*indices2].plot(density_array, array(Delta_s_GA_U2_td2) / V1, ls_GA_s, linewidth=lw)
        ax[*indices2].plot(density_array, array(Delta_d_GA_U2_td2) / V1, ls_GA_d, linewidth=lw)
        ax[*indices2].plot(density_array, array(abs(Delta_s_s_id_GA_U2_td2 / V1 + 1j * Delta_d_s_id_GA_U2_td2 / V1)),
                      ls_GA_s_id, linewidth=lw)
        ax[*indices2].plot(density_array, array(Delta_s_HF_U2_td2) / V1, ls_HF_s, linewidth=lw, alpha=opaci_HF)
        ax[*indices2].plot(density_array, array(Delta_d_HF_U2_td2) / V1, ls_HF_d, linewidth=lw, alpha=opaci_HF)
        ax[*indices2].plot(density_array, array(abs(Delta_s_id_HF_U2_td2 / V1 + 1j * Delta_d_s_id_HF_U2_td2 / V1)),
                      ls_HF_s_id, linewidth=lw, alpha=opaci_HF)
        ax[*indices2].grid()


        if plot_also_cidciu:
            ax[1, 0].plot(density_array, cidciu_s_GA_U1_td2, ls_GA_s, linewidth=lw)
            ax[1, 0].plot(density_array, cidciu_d_GA_U1_td2, ls_GA_d, linewidth=lw)
            ax[1, 0].plot(density_array, cidciu_s_id_GA_U1_td2, ls_GA_s_id, linewidth=lw)
            ax[1, 0].plot(density_array, cidciu_s_HF_U1_td2, ls_HF_s, linewidth=lw, alpha=opaci_HF)
            ax[1, 0].plot(density_array, cidciu_d_HF_U1_td2, ls_HF_d, linewidth=lw, alpha=opaci_HF)
            ax[1, 0].plot(density_array, cidciu_s_id_HF_U1_td2, ls_HF_s_id, linewidth=lw, alpha=opaci_HF)
            ax[1, 0].set_ylabel(r'$\langle c_{i\downarrow} c_{i{\uparrow}} \rangle$', fontsize=fs)
            ax[1, 0].set_xlabel(r'electron density $n$', fontsize=fs)
            ax[1, 0].grid()

            ax[1, 1].plot(density_array, cidciu_s_GA_U2_td2, ls_GA_s, linewidth=lw)
            ax[1, 1].plot(density_array, cidciu_d_GA_U2_td2, ls_GA_d, linewidth=lw)
            ax[1, 1].plot(density_array, cidciu_s_id_GA_U2_td2, ls_GA_s_id, linewidth=lw)
            ax[1, 1].plot(density_array, cidciu_s_HF_U2_td2, ls_HF_s, linewidth=lw, alpha=opaci_HF)
            ax[1, 1].plot(density_array, cidciu_d_HF_U2_td2, ls_HF_d, linewidth=lw, alpha=opaci_HF)
            ax[1, 1].plot(density_array, cidciu_s_id_HF_U2_td2, ls_HF_s_id, linewidth=lw, alpha=opaci_HF)
            #ax[1, 1].legend(['GA - s', 'GA - d', 'GA - s+id', 'HF - s', 'HF - d', 'HF - s+id'], fontsize=fs)
            ax[1, 1].set_xlabel(r'electron density $n$', fontsize=fs)
            ax[1, 1].grid()

        savefig('poster_plots_td2.png')

        show()

    else:
        if GA:
            folder_name = 'results_with_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            folder_name_HF = 'results_no_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            path_HF = './' + folder_name_HF + '/'
            str_part_HF = '_no_GA'
        else:
            folder_name = 'results_no_GA' + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)

        path = './' + folder_name + '/'
        fs = 15

        str_part = '_with_GA' if GA else '_no_GA'

        Delta_s = loadtxt(os.path.join(path, 'res_Delta_s' + str_part))
        Delta_d = loadtxt(os.path.join(path, 'res_Delta_d' + str_part))
        if plot_s_id:
            Delta_s_s_id = loadtxt(os.path.join(path, 'res_Delta_s_s_id' + str_part))
            Delta_d_s_id = loadtxt(os.path.join(path, 'res_Delta_d_s_id' + str_part))

        E_s = loadtxt(os.path.join(path, 'res_E_s' + str_part))
        E_d = loadtxt(os.path.join(path, 'res_E_d' + str_part))
        if plot_s_id:
            E_s_id = loadtxt(os.path.join(path, 'res_E_s_id' + str_part))
        cidciu_s = loadtxt(os.path.join(path, 'res_cidciu_s' + str_part))
        cidciu_d = loadtxt(os.path.join(path, 'res_cidciu_d' + str_part))

        if GA:
            cidciu_s_HF = loadtxt(os.path.join(path_HF, 'res_cidciu_s' + str_part_HF))
            cidciu_d_HF = loadtxt(os.path.join(path_HF, 'res_cidciu_d' + str_part_HF))
            cidciu_s_id_HF = loadtxt(os.path.join(path_HF, 'res_cidciu_s_id' + str_part_HF))

        if plot_s_id:
            cidciu_s_id = loadtxt(os.path.join(path, 'res_cidciu_s_id' + str_part))

        density_s = loadtxt(os.path.join(path, 'res_density_s' + str_part))
        density_d = loadtxt(os.path.join(path, 'res_density_d' + str_part))
        if plot_s_id:
            density_s_id = loadtxt(os.path.join(path, 'res_density_s_id' + str_part))



        if GA:
            fig, ax = subplots(1,3, figsize = (12, 6))
        else:
            fig, ax = subplots(1,3, figsize = (12, 6))
        fig.tight_layout(pad=4.0)

        if GA:
            fig.suptitle('With GA, U = ' + str(U) + ', t\' =' + str(t_diag) + ', V1 = ' + str(V1)  + ', kBT =' + str(kBT), fontsize = 16)
        else:
            fig.suptitle('No GA, U = ' + str(U) + ', t\' =' + str(t_diag) + ', V1 = ' + str(V1) + ', kBT ='+ str(kBT), fontsize = 16)



        if not plot_s_id:
            # plot Delta_s, Delta_d and Delta_s_id
            ax[0].plot(density_s, array(Delta_s)/V1, '--b.')
            ax[0].plot(density_d, array(Delta_d)/V1, ':r.')

            #ax[0].legend()
            ax[0].set_xlabel('electron density $n$', fontsize = fs)
            #ax[0,0].set_ylabel('gap $\Delta$/$V_1$')
            ax[0].set_ylabel(r'$\Delta/V_1 = \langle c_{i+\delta \downarrow} c_{i \uparrow} \rangle$', fontsize = fs)
            ax[0].legend(['s-symmetry', 'd-symmetry'])
            ax[0].grid()

            # plot cidciu_s, cidciu_d and cidciu_s_id
            if GA:
                ax[1].plot(density_s, cidciu_s, '--b.')
                ax[1].plot(density_d, cidciu_d, ':r.')
            else:
                ax[1].plot(density_s, cidciu_s, '--b.')
                ax[1].plot(density_d, cidciu_d, ':r.')

            ax[1].set_xlabel('electron density $n$', fontsize = fs)
            #ax[0,1].set_ylabel(r'$\langle c_{i\downarrow} c_{i{\uparrow}} \rangle$')
            ax[1].legend(['s-symmetry', 'd-symmetry'])
            if GA:
                ax[1].set_ylabel(r'$J_x = \langle c_{i\downarrow} c_{i{\uparrow}} \rangle$', fontsize=fs)
            else:
                ax[1].set_ylabel(r'$L/U = \langle c_{i\downarrow} c_{i\uparrow} \rangle$', fontsize=fs)
            #ax[1].set_ylabel(r'$L/U = \langle c_{i\downarrow} c_{i\uparrow} \rangle$', fontsize = fs)
            ax[1].grid()

            ax[2].plot(density_s, E_s-E_d, '-k.')
            #ax[0,2].plot(density_d, E_d, ':r.')
            #ax[0,2].plot(density_d, E_s_id, '-.k.')
            #ax[0,2].set_ylabel('Energy per grid point E/N')

            #ax[0,2].set_title('Energy per grid point E/N')
            ax[2].set_ylabel(r'$(E_s - E_d)/N$', fontsize = fs)
            ax[2].set_xlabel('electron density $n$', fontsize = fs)
            ax[2].grid()
            #ax[0,2].plot(density_s, E_s-E_d, '-k.')


        else:
            ax[0].plot(density_s, array(abs(Delta_s_s_id / V1 + 1j* Delta_d_s_id / V1)), '.-g')
            ax[0].plot(density_s, array(Delta_s_s_id) / V1, 'k', label = r'$s+id: \Delta_s$', linewidth = 3, alpha = 0.3)
            ax[0].plot(density_d, array(Delta_d_s_id) / V1, 'm', label = r'$s+id: \Delta_d$', linewidth = 3, alpha = 0.3)
            ax[0].legend([r'$|\Delta_x|/V1$', r'$\Delta_s/V1$', r'$\Delta_d/V1$'])
            ax[0].set_xlabel('electron density $n$', fontsize=fs)
            ax[0].set_ylabel('gap', fontsize=fs)
            ax[0].grid()

            if GA:
                ax[1].plot(density_s, cidciu_s_id, '.-g')
            else:
                ax[1].plot(density_s, cidciu_s_id, '.-g')
            ax[1].set_xlabel('electron density $n$', fontsize=fs)
            if GA:
                ax[1].set_ylabel(r'$J_x = \langle c_{i\downarrow} c_{i{\uparrow}} \rangle$', fontsize=fs)
            else:
                ax[1].set_ylabel(r'$L/U = \langle c_{i\downarrow} c_{i\uparrow} \rangle$', fontsize=fs)
            ax[1].grid()

            if GA:
                ax[2].plot(density_s, E_s_id, '.-g')
                ax[2].set_ylabel('Energy per grid point E/N', fontsize=fs)
            else:
                ax[2].plot(density_s, E_s_id, '.-g')
                ax[2].set_ylabel('Energy per grid point E/N', fontsize=fs)
            ax[2].set_xlabel('electron density $n$', fontsize=fs)

            ax[2].grid()





        # Shrink current axis's height by 10% on the bottom
        #box = ax[1].get_position()
        #ax[1].set_position([box.x0, box.y0 + box.height * 0.1,
        #                 box.width, box.height * 0.9])


        if GA:
            if plot_s_id:
                name_for_saving = 's_id_With_GA_' + 'U = ' + str(U) + ', t\' =' + str(t_diag) + ', V1 = ' + str(V1)
            else:
                name_for_saving = 'With_GA_' + 'U = ' + str(U) +', t\' =' + str(t_diag) + ', V1 = ' + str(V1)
        else:
            if plot_s_id:
                name_for_saving = 's_id_No_GA_' + 'U = ' + str(U) + ', t\' =' + str(t_diag) + ', V1 = ' + str(V1)
            else:
                name_for_saving = 'No_GA_' + 'U = ' + str(U) + ', t\' =' + str(t_diag) + ', V1 = ' + str(V1)
        savefig(os.path.join(folder_name, name_for_saving + '.png'))


        show()


        HF_D = lambda n: n**2/4.0
        # plot of the doubly occupied sites
        if GA and not plot_s_id:
            HF_D_arr_s = HF_D(density_s) + cidciu_s_HF**2
            HF_D_arr_d = HF_D(density_d) + cidciu_d_HF**2
            fig, ax = subplots(1,2, figsize = (12, 6))
            D_s = loadtxt(os.path.join(path, 'res_D_s' + str_part))
            D_d = loadtxt(os.path.join(path, 'res_D_d' + str_part))


            ax[0].plot(density_s, D_s, '-b.')
            ax[0].plot(density_d, D_d, '-r.')
            ax[0].plot(density_s, HF_D_arr_s, '--b')
            ax[0].plot(density_d, HF_D_arr_d, '--r')
            ax[0].set_xlabel(r'electron density $n$', fontsize = fs)
            ax[0].set_ylabel(r'$D$', fontsize = fs)
            ax[0].grid()
            ax[0].legend([r'GA: $D_s$', r'GA: $D_d$', r'HF: $D_s$', r'HF: $D_d$'])
            ax[0].set_title('Double occupancy D', fontsize = fs)

            # plot of the lamda_d
            lamda_s = loadtxt(os.path.join(path, 'res_lamda_s' + str_part))
            lamda_d = loadtxt(os.path.join(path, 'res_lamda_d' + str_part))
            for j, el in enumerate(lamda_s):
                if el < 1.0e-7:
                    lamda_s[j] = 0.0
            for j, el in enumerate(lamda_d):
                if el < 1.0e-7:
                    lamda_d[j] = 0.0
            ax[1].plot(density_s, lamda_s, '-b.')
            ax[1].plot(density_d, lamda_d, '-r.')
            ax[1].set_xlabel(r'electron density $n$', fontsize = fs)
            ax[1].set_ylabel(r'$\lambda$', fontsize = fs)
            ax[1].grid()
            ax[1].legend([r'$\lambda_s$', r'$\lambda_d$'])
            ax[1].set_title(r'Lagrange multiplier $\lambda$', fontsize = fs)

            name_for_saving_D = 'D_lambda_plot' + str_part + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            savefig(os.path.join(folder_name, name_for_saving_D + '.png'))
            show()

            ZZ_s = loadtxt(os.path.join(path, 'res_ZZ_s' + str_part))
            ZZ_d = loadtxt(os.path.join(path, 'res_ZZ_d' + str_part))
            fig, ax = subplots(1,1, figsize = (6, 6))
            ax.plot(density_s, ZZ_s, '-b.', density_d, ZZ_d, '-r.')
            ax.set_xlabel(r'electron density $n$', fontsize = fs)
            ax.set_ylabel(r'$Z$', fontsize = fs)
            ax.grid()
            ax.legend([r'$Z_s$', r'$Z_d$'])
            ax.set_title('Z', fontsize = fs)
            name_for_saving_Z = 'Z_plot' + str_part + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            savefig(os.path.join(folder_name, name_for_saving_Z + '.png'))
            show()



        if GA and plot_s_id:
            fig, ax = subplots(1,2, figsize = (12, 6))
            HF_D_arr_s = HF_D(density_s_id) + cidciu_s_id_HF**2

            # plot of the doubly occupied sites
            D_s_id = loadtxt(os.path.join(path, 'res_D_s_id' + str_part))
            ax[0].plot(density_s, D_s_id, '-g.')
            ax[0].plot(density_s, HF_D_arr_s, '--k')
            ax[0].set_xlabel(r'electron density $n$', fontsize=fs)
            ax[0].set_ylabel(r'$D$', fontsize=fs)
            ax[0].grid()
            ax[0].legend([r'GA:  $D_{s+id}$', r'HF: $D_{s+id}$'])
            ax[0].set_title('Double occupancy D', fontsize=fs)


            lamda_s_id = loadtxt(os.path.join(path, 'res_lamda_s_id' + str_part))
            for j, el in enumerate(lamda_s_id):
                if el < 1.0e-7:
                    lamda_s_id[j] = 0.0
            ax[1].plot(density_s, lamda_s_id, '-g.')
            ax[1].set_xlabel(r'electron density $n$', fontsize = fs)
            ax[1].set_ylabel(r'$\lambda$', fontsize = fs)
            ax[1].grid()
            ax[1].set_title(r'Lagrange multiplier $\lambda$', fontsize=fs)


            name_for_saving_D = 's_id_D_lambda_plot' + str_part + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            savefig(os.path.join(folder_name, name_for_saving_D + '.png'))
            show()

            ZZ_s_id = loadtxt(os.path.join(path, 'res_ZZ_s_id' + str_part))
            fig, ax = subplots(1, 1, figsize=(6, 6))
            ax.plot(density_s, ZZ_s_id, '-g.')
            ax.set_xlabel(r'electron density $n$', fontsize=fs)
            ax.set_ylabel(r'$Z$', fontsize=fs)
            ax.grid()
            ax.legend([r'$Z_s$', r'$Z_d$'])
            ax.set_title('Z', fontsize=fs)
            name_for_saving_Z = 'Z_plot' + str_part + '_U_' + str(U) + '_t_diag_' + str(t_diag) + '_V1_' + str(V1)
            savefig(os.path.join(folder_name, name_for_saving_Z + '.png'))
            show()



