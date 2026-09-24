import numpy as np
from scipy.interpolate import interp1d, interp2d
from matplotlib.colors import LinearSegmentedColormap

def map_to_screen(p,d1,d2,home_loc):
    qx = p[0] * (d1/2)
    qy = p[1] * (d2/2) if home_loc == "center" else p[1]*d2    
    # if p[1] < 0: qy = 0
    return qx, -qy

def bound_cursor(cursor_pos,r_cursor,d1,d2):
    """ Bound cursor position to screen limits """
    px, py = cursor_pos[0],cursor_pos[1]
    px = max(min(px, d1 - r_cursor * 2), r_cursor * 2)
    py = max(min(py, d2 - r_cursor * 2), r_cursor * 2)
    cursor_pos = np.array([px,py])
    return cursor_pos

def map_body_to_cursor(q,d1,d2,home_loc,gains,offset):
    """ get sensor position command q: [-1,1] and map to screen
        multiplying by gains if needed """
    sx, sy = map_to_screen(q,d1,d2,home_loc)
    sx *= gains[0] if sx < 0 else gains[1]
    sy *= gains[2] if sy > 0 else gains[3]

    desPos = np.array([sx, sy]) + offset
    return desPos

def resample_traj(time_ref,trajectories,fs_new):
    total_duration = time_ref[-1] - time_ref[0]
    num_samples_new = int(total_duration * fs_new) + 1
    time_ref_new = np.linspace(time_ref[0], time_ref[-1], num_samples_new)

    num_trajectories = trajectories.shape[0]
    traj_resampled = np.zeros((num_trajectories,num_samples_new,2))  # Initialize resampled trajectory
    for k in range(num_trajectories):
        traj = trajectories[k,:,:]
        for i in range(2):  # Iterate over dimensions of the trajectory (e.g., x, y)
            interpolator = interp1d(time_ref, traj[:, i], kind='linear')
            traj_resampled[k,:,i] = interpolator(time_ref_new)
    return traj_resampled, time_ref_new

def get_home_center(curr_traj,r_circle, m=5):
    got_it = False
    while got_it == False:
        try:
            vec = curr_traj[0] - curr_traj[m]
            vec_norm = vec / (1e-5 + np.linalg.norm(vec))
            got_it = True
        except:
            m += 2           
    p_home = curr_traj[0] - (r_circle + 1) * (vec_norm).astype(int)
    return tuple(p_home.astype(int))

def create_gradient(start_color, end_color, name):
    """
    Create a gradient colormap from start_color to end_color with the given name.
    """
    cmap_data = [(0, start_color),(1, end_color)]
    return LinearSegmentedColormap.from_list(name, cmap_data)

def muscle_color_grad(num_colors):
    color_gradients = {
        "blue_to_purple": ((0, 0, 1), (0.5, 0, 0.5)),
        "yellow_to_orange": ((1, 0.65, 0),(1, 1, 0)),
        "green_to_dark_green": ((0, 0.5, 0), (1, 1, 0)),
        "red_to_pink": ((1, 0, 0), (1, 0.75, 0.79))
    }
    values = np.linspace(0, 1, num_colors)

    gradients = []
    for gradient_name, (start_color, end_color) in color_gradients.items():
        gradient_cmap = create_gradient(start_color, end_color, gradient_name)
        gradients.append(gradient_cmap(values))

    return gradients


def normalize_time_series(time_series):
    """
    Normalize time series to represent the percentage of movement.
    """
    # Normalize each time series by dividing by the difference
    normalized_series = (time_series - time_series[0]) / (time_series[-1] - time_series[0])
    normalized_series *= 100

    return normalized_series

def resample_data(og_time, og_data, new_time):
    new_data = np.zeros((len(new_time),og_data.shape[-1]))
    for j in range(og_data.shape[-1]):
        new_data[:,j] = np.interp(new_time, og_time, og_data[:,j])
    return new_data