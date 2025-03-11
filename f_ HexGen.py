import matplotlib.pyplot as plt
import math
import numpy as np
from hexalattice.hexalattice import plot_single_lattice
from _utils import point_in_circle
import matplotlib.patches as patches
from _utils import _beam_align_BS_

def plot_hexagonal_network(n_row=4,n_col=4,hex_size=20,seed=0,R_min=0.2,
                           R_max=0.8, beam_divisor=6, beam_len=1.0,
                           SaveData=False,save_dir='./data/position/'):
    r"""plot hexagon network with n_row*n_col BSs and UEs
     Input:
        n_row: num of rows
        n_col: num of columns
        hex_size: length of hexagon edge
        seed: UE position random seed
        R_min: min distance to BS (unit: hex_size)
        R_max: max distance to BS (unit: hex_size)
        beam_divisor: type: int, beamwidth = pi/beam_divisor
        beam_len: length of beam wedge (unit: hex_size)
        SaveData: bool, overwrite existing position data if True
        save_dir: directory to save BS/UE positions
    Output:
        BS_position: size=(n_row*n_col,2), BS positions
        UE_positions: size =(n_row*n_col,2), UE positions
        save figure, save data
        """
    n_BS, n_UE = n_row*n_col, n_row*n_col
    BS_position = np.zeros([n_BS,2])
    """compute BS positions"""
    for i in range(n_BS):
        if math.floor(i/n_col)%2==0:
            BS_position[i,0] = (i%n_col)*math.sqrt(3)*hex_size+math.sqrt(3)/2*hex_size
            BS_position[i,1] = hex_size+hex_size*math.floor(i/n_col)*3/2
        else:
            BS_position[i,0] = (i%n_col+1)*math.sqrt(3)*hex_size
            BS_position[i,1] = hex_size*5/2+math.floor(math.floor(i/n_col)/2)*3*hex_size
    x_BS,y_BS = BS_position[:,0], BS_position[:,1]
    """ generate hexagons centered at BSs """
    _, ax = plt.subplots()
    h_ax = plot_single_lattice(coord_x=x_BS,
                               coord_y=y_BS,
                               face_color="w",
                               edge_color="k",
                               min_diam=np.sqrt(3)*hex_size,  # dist between two parallel edges
                               plotting_gap=0,
                               rotate_deg=0,
                               h_ax=ax)  # pass the subplots handle in
    """ plot hexagons """
    h_ax.plot(x_BS, y_BS, " ", marker="o", markersize=4,
              markeredgecolor="k", markerfacecolor="k",label="BS")  # BSs
    """draw circles around each BS"""
    for i in range(n_BS):
        circle = plt.Circle((x_BS[i],y_BS[i]),radius=R_min*hex_size,fill=False,ls=":")
        # circle_ = plt.Circle((x_BS[i], y_BS[i]), radius=R_max * hex_size, fill=False, ls=":")
        h_ax.add_artist(circle)
        # h_ax.add_artist(circle_)
    """label BSs"""
    offset=5
    for bs in range(n_BS):
        h_ax.text(BS_position[bs,0]+offset,BS_position[bs,1]+offset,f"BS {bs}")
    """generate UEs"""
    UE_position = np.zeros([n_BS, 2])
    UE_position[:,0],UE_position[:,1]=point_in_circle(x=x_BS,y=y_BS,r_min=R_min*hex_size,
                                                      r_max=R_max*hex_size,seed=seed)
    if SaveData:  #save position data
        np.save(save_dir+'BS_position_small',BS_position)
        np.save(save_dir+'UE_position_small',UE_position)
    """ plot UEs """
    h_ax.plot(UE_position[:,0],UE_position[:,1], " ",marker="o",markersize=3,
              markeredgecolor="b",markerfacecolor="b",label="UE")
    """ Add beams"""
    wedge_len = beam_len*hex_size  #beam wedge length
    UE_scheduled = 0 * np.ones(n_BS,dtype=np.int32)
    beamwidth = np.pi / beam_divisor
    Beam_direction_BS = _beam_align_BS_(BS_position=BS_position,UE_position=UE_position,
                                        Scheduled_UE=UE_scheduled,beamwidth=beamwidth)
    for i in range(n_BS):
        ax.add_patch(patches.Wedge((x_BS[i],y_BS[i]),wedge_len,Beam_direction_BS[i,0]*(180/np.pi),
                                   Beam_direction_BS[i, 1]*(180/np.pi),color="green",alpha=0.5))
    ax.set_xlabel("x axis (meters)")
    ax.set_ylabel("y axis (meters)")
    ax.legend()
    plt.savefig("./figure/fig_position.pdf")
    plt.show()





if __name__ == "__main__":
    plot_hexagonal_network(n_row=2,
                           n_col=2,
                           hex_size=50,
                           seed=11,
                           beam_divisor=3,
                           SaveData= False,
                           save_dir='./data/position/')




