"""
This file contains necessary utility functions for the assignment. 
Please do not modify the code in this file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import inv, det
from scipy.spatial.distance import cdist

def generate_data(gp_field, field_range, field_resolution, num_agents, num_samples_per_agent, noise_std):
    """
    Generates noisy samples from the true GP field and uniformly allocates them to agents.
    
    Args:
        gp_field (numpy.ndarray): 2D array representing the ground-truth field.
        field_range (numpy.ndarray): Array of shape (2, 2) defining the [min, max] bounds 
                                     for the x1 and x2 coordinates.
        field_resolution (int): Integer defining the grid resolution of the field.
        num_agents (int): Total number of agents in the network.
        num_samples_per_agent (int): Number of samples allocated to each agent.
        noise_std (float): Standard deviation of the additive Gaussian measurement noise.
        
    Returns:
        sample_locations_reshaped (numpy.ndarray): Array of shape (GP_INPUT_DIM, num_samples_per_agent, 
                                                   num_agents) containing the distributed locations.
        sample_locations_list (list): List of arrays containing the distributed locations.
        noisy_samples_reshaped (numpy.ndarray): Array of shape (num_samples_per_agent, num_agents) 
                                                containing the distributed noisy measurements.
        sample_locations (numpy.ndarray): Array of shape (GP_INPUT_DIM, total_num_samples) 
                                          containing all generated locations.
        clean_samples (numpy.ndarray): Array of shape (total_num_samples,) containing the true 
                                       field values at the sample locations.
        mesh_x1 (numpy.ndarray): 2D array representing the x1 grid for plotting.
        mesh_x2 (numpy.ndarray): 2D array representing the x2 grid for plotting.
    """

    # generate samples locations [GP_INPUT_DIM, TOTAL_NUM_SAMPLES]
    total_num_samples = num_agents * num_samples_per_agent
    ################# YOUR CODE HERE #################
    sample_locations = np.array([np.random.uniform(field_range[0,0], field_range[0,1], total_num_samples),
                                np.random.uniform(field_range[1,0], field_range[1,1], total_num_samples)])
    ##################################################

    clean_samples, mesh_x1, mesh_x2 = data_generator(sample_locations, gp_field, field_range, field_resolution)

    # generate noisy samples
    ################# YOUR CODE HERE #################
    sample_noise = np.random.normal(0, noise_std, total_num_samples)
    noisy_samples = clean_samples + sample_noise
    ##################################################

    # uniformly allocated samples to agents
    noisy_samples_reshaped = np.reshape(noisy_samples, (num_samples_per_agent,num_agents), order = 'C')
    # clean_samples_reshaped = np.reshape(clean_samples, (num_samples_per_agent,num_agents), order = 'C')
    sample_locations_reshaped = np.reshape(sample_locations, (2, num_samples_per_agent,num_agents), order = 'C')

    sample_locations_list = np.split(sample_locations_reshaped, sample_locations_reshaped.shape[2], axis=2)
    sample_locations_list = [arr.squeeze(axis=2) for arr in sample_locations_list]
    
    return sample_locations_reshaped, sample_locations_list, noisy_samples_reshaped, sample_locations, clean_samples, mesh_x1, mesh_x2


def compute_gp_gradient(eval_point_dict, sample_locations, noisy_samples, sample_size, input_dim=2):
    """
    Computes the gradient of the Negative Log-Likelihood evaluated at a specific hyperparameter point.
    """
    gradients = {}
    
    # Build covariance matrices using the evaluation point
    K_n = eval_point_dict['noise_std']**2 * np.eye(sample_size)

    scaled_sample_locations = np.copy(np.diag(1 / eval_point_dict['length_scale']) @ sample_locations)
    squared_distance_mat = cdist(scaled_sample_locations.T, scaled_sample_locations.T, 'sqeuclidean')
    
    K_s = eval_point_dict['signal_amplitude']**2 * np.exp(-0.5 * squared_distance_mat)
    K = K_s + K_n

    K_inv = inv(K + 1e-10 * np.eye(np.shape(K)[0]))
    constant_1 = K_inv - K_inv @ np.outer(noisy_samples, noisy_samples) @ K_inv.T
    
    # Calculate NLL objective
    objective = noisy_samples.T @ K_inv @ noisy_samples + np.log(det(K))

    # Gradient w.r.t signal_amplitude
    K_div_signal_amplitude = 2 / eval_point_dict['signal_amplitude'] * K_s
    partial_signal_amplitude = 0.5 * np.trace(constant_1 @ K_div_signal_amplitude)

    # Gradient w.r.t length_scale
    partial_length_scale = np.zeros(input_dim)
    squared_distance_dim_1 = cdist(sample_locations[[0],:].T, sample_locations[[0],:].T, 'sqeuclidean')
    squared_distance_dim_2 = cdist(sample_locations[[1],:].T, sample_locations[[1],:].T, 'sqeuclidean')

    K_div_l_dim_1 = squared_distance_dim_1 * K_s * eval_point_dict['length_scale'][0]**(-3)
    K_div_l_dim_2 = squared_distance_dim_2 * K_s * eval_point_dict['length_scale'][1]**(-3)
    
    partial_length_scale[0] = 0.5 * np.trace(constant_1 @ K_div_l_dim_1)
    partial_length_scale[1] = 0.5 * np.trace(constant_1 @ K_div_l_dim_2)

    # Gradient w.r.t noise_std
    K_div_noise_std = 2 * np.sqrt(K_n)
    partial_noise_std = 0.5 * np.trace(constant_1 @ K_div_noise_std)

    # Store in dict
    gradients['partial_signal_amplitude'] = partial_signal_amplitude
    gradients['partial_l'] = partial_length_scale
    gradients['partial_noise_std'] = partial_noise_std

    return gradients, objective


def is_graph_connected(adjacency_matrix):

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Compute eigenvalues of the Laplacian matrix
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)

    # Count zero eigenvalues
    zero_eigenvalues = np.sum(np.isclose(eigenvalues, 0))

    # Graph is connected if there is exactly one zero eigenvalue
    return zero_eigenvalues == 1


def param_array_to_dict(param_array):
    """
    Converts a 1D numpy array to a formated dictionary of hyperparameters
    """
    return {
        'signal_amplitude': param_array[0],
        'length_scale': np.array([param_array[1], param_array[2]]),
        'noise_std': param_array[3]
    }

def param_dict_to_array(param_dict):
    """
    Converts a dictionary of hyperparameters to a 1D numpy array
    """

    return np.array([param_dict['signal_amplitude'], param_dict['length_scale'][0], param_dict['length_scale'][1], param_dict['noise_std']])


def data_generator(sample_locations, gp_field, field_range, field_resolution):
    """
    Generates clean samples from the GP field at the given sample locations
    - sample_locations: 2D numpy array of shape (INPUT_DIM, NUM_SAMPLES)
    """

    x1 = np.linspace(field_range[0,0], field_range[0,1], field_resolution[0])
    x2 = np.linspace(field_range[1,0], field_range[1,1], field_resolution[1])

    interp_func = RegularGridInterpolator((x1, x2), gp_field)
    clean_samples = interp_func(sample_locations.T)

    mesh_x1, mesh_x2 = np.meshgrid(x1, x2, indexing='ij')

    return clean_samples, mesh_x1, mesh_x2

def visualize_samples_on_field(ax, mesh_x1, mesh_x2, gp_field, sample_locations, samples):
    """
    Visualizes the GP field and the samples
    """
    # plt.figure(figsize=(8, 6))

    # Plot the original grid data
    contour = ax.contourf(mesh_x1, mesh_x2, gp_field, levels=50, cmap='plasma', alpha=0.7)
    # ax.colorbar(label='Field Value')

    # Overlay agents and sample locations and their interpolated values
    ax.scatter(sample_locations[0,:], sample_locations[1, :], c=samples, s=50, cmap='plasma', edgecolor='k', label='Sample Points')

    # Labels and title
    ax.set_title("GP Field")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # plt.legend(loc='upper right')
    # plt.show()
    return contour

def visualize_sample_allocation(ax, agent_locations, sample_locations_list: list, adjacency_matrix):
    """
    Visualizes the sample allocation among agents
    """
    num_agents = agent_locations.shape[1]
    
    # Note: plt.get_cmap is the modern equivalent to plt.cm.get_cmap
    colors = plt.get_cmap("rainbow", num_agents)  

    # Plot the agents
    for i in range(num_agents):
        # Plot agent centers (No label here, so we don't get duplicate legend entries)
        ax.scatter(agent_locations[0, i], agent_locations[1, i], marker='x', s=70, color=colors(i))
        
        # Plot samples AND add the label here for the legend
        ax.scatter(sample_locations_list[i][0,:], sample_locations_list[i][1,:], 
                   marker='o', color=colors(i), label=f'Agent {i}')

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                ax.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1)
    
    ax.set_aspect("equal")

def visualize_graphs(agent_locations , adjacency_matrix):
    """
    Visualizes the underlying graph
    """
    num_agents = agent_locations.shape[1]
    colors = plt.cm.get_cmap("tab10", num_agents)  # Use a colormap with N unique colors

    # Plot the agents
    plt.figure(figsize=(3, 3))
    for i in range(num_agents):
        plt.scatter(agent_locations[0, i], agent_locations[1, i], marker = 'x', s = 70, color=colors(i))
        # plt.scatter(sample_locations_list[i][0,:], sample_locations_list[i][1,:], marker = 'o', color=colors(i))

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                plt.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1)
    
    plt.axis("equal")
    plt.show()
