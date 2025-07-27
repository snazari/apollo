import graph_tool.all as gt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from matplotlib import colormaps
import os
import subprocess

def run_french_degroot_dynamics(g: gt.Graph,
                                initial_opinions: np.ndarray = None,
                                max_iters: int = 100,
                                stubbornness: float = 0.0):
    """
    Runs the French-DeGroot opinion dynamics simulation on a graph-tool graph.
    This function only performs the simulation and returns the data.
    """
    num_nodes = g.num_vertices()
    if initial_opinions is None:
        opinions = np.random.rand(num_nodes)
    else:
        opinions = initial_opinions.copy()

    opinion_history = [opinions.copy()]

    adj_matrix = gt.adjacency(g).toarray()
    identity_matrix = np.identity(num_nodes)
    influence_matrix = (1 - stubbornness) * adj_matrix + stubbornness * identity_matrix
    row_sums = influence_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    trust_matrix = influence_matrix / row_sums[:, np.newaxis]

    for i in range(max_iters):
        new_opinions = trust_matrix @ opinions
        opinion_history.append(new_opinions.copy())
        if np.allclose(new_opinions, opinions):
            print(f"Converged after {i+1} iterations.")
            break
        opinions = new_opinions

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}) without full convergence.")

    history_df = pd.DataFrame(opinion_history, columns=[f'Node_{v}' for v in g.vertices()])
    history_df.index.name = 'Time_Step'

    return history_df, opinions, adj_matrix, influence_matrix, trust_matrix

def create_progression_snapshots(g, history_df, num_snapshots=5, output_dir="opinion_snapshots"):
    """
    Generates a series of PNG snapshots showing the opinion progression.
    """
    print(f"\nGenerating {num_snapshots} opinion progression snapshots in '{output_dir}'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_steps = len(history_df) - 1
    # Ensure we get unique, evenly spaced indices, including first and last
    indices = np.unique(np.linspace(0, total_steps, num_snapshots, dtype=int))

    cmap = colormaps.get_cmap('plasma')
    color_map = mcolors.LinearSegmentedColormap.from_list("custom_map", cmap.colors)
    v_opinions = g.new_vertex_property("double")

    # Normalize color map consistently across all frames from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i in indices:
        row = history_df.iloc[i]

        # For the final snapshot, force a single consensus color
        if i == total_steps:
            consensus_value = row.values.mean()
            v_opinions.a = np.full(g.num_vertices(), consensus_value)
        else:
            v_opinions.a = row.values

        output_filename = os.path.join(output_dir, f"opinion_progression_step_{i:04d}.png")
        
        gt.graph_draw(g,
                      vertex_fill_color=v_opinions, vertex_color=v_opinions,
                      vcmap=color_map, vertex_size=15, edge_pen_width=1.5,
                      output_size=(800, 800), output=output_filename)
        print(f"Saved snapshot: {output_filename}")

    print("All snapshots generated.")

def create_opinion_timeseries_plots(history_df, nodes_per_plot=10):
    """
    Generates Plotly timeseries plots of the opinion history.
    """
    print(f"\nGenerating opinion timeseries plots...")
    node_columns = history_df.columns
    num_plots = (len(node_columns) + nodes_per_plot - 1) // nodes_per_plot

    for i in range(num_plots):
        start_index = i * nodes_per_plot
        end_index = start_index + nodes_per_plot
        nodes_to_plot = node_columns[start_index:end_index]

        fig = go.Figure()
        for node in nodes_to_plot:
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df[node], 
                                     mode='lines', name=node))

        fig.update_layout(
            title=f"Opinion Dynamics History (Nodes {start_index}-{end_index-1})",
            xaxis_title="Time Step",
            yaxis_title="Opinion Value",
            legend_title="Node ID"
        )
        output_filename = f"opinion_timeseries_{i+1}.html"
        fig.write_html(output_filename)
        print(f"Saved timeseries plot: {output_filename}")

def create_matrix_heatmap(matrix, title, output_filename):
    """
    Generates and saves a Plotly heatmap for a given matrix.
    """
    print(f"\nGenerating heatmap for '{title}'...")
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale='Viridis'))
    fig.update_layout(
        title=title,
        xaxis_title="Node Index",
        yaxis_title="Node Index",
        yaxis_autorange='reversed'  # This places (0,0) in the upper-left corner
    )
    fig.write_html(output_filename)
    print(f"Heatmap saved to '{output_filename}'")

def calculate_and_display_social_power(trust_matrix):
    """
    Calculates and visualizes the social power of each node.
    """
    print("\n--- Calculating Social Power ---")
    # Social power is the left eigenvector of the trust matrix for eigenvalue 1.
    # This is equivalent to the right eigenvector of the transposed matrix.
    eigenvalues, eigenvectors = np.linalg.eig(trust_matrix.T)
    
    # Find the eigenvector corresponding to the eigenvalue closest to 1
    power_vector_index = np.argmin(np.abs(eigenvalues - 1))
    power_vector = np.real(eigenvectors[:, power_vector_index])
    
    # Normalize the vector to sum to 1
    power_vector /= power_vector.sum()
    
    print("Social Power Vector (influence on consensus):")
    for i, power in enumerate(power_vector):
        print(f"  Node {i}: {power:.4f}")

    # Create a bar chart visualization
    node_ids = [f'Node_{i}' for i in range(len(power_vector))]
    fig = go.Figure([go.Bar(x=node_ids, y=power_vector)])
    fig.update_layout(
        title="Social Power of Each Node",
        xaxis_title="Node ID",
        yaxis_title="Normalized Social Power",
        xaxis_tickangle=-45
    )
    fig.write_html("social_power_barchart.html")
    print("\nSocial power bar chart saved to 'social_power_barchart.html'")

def create_visualizations(g, history_df, final_opinions, trust_matrix):
    """
    Handles the creation of all visualizations from the simulation results.
    """
    #print("\n--- Opinion Dynamics History ---")
    #with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):
    #    print(history_df)

    print("\nGenerating Plotly table of opinion history...")
    history_df_rounded = history_df.round(4)
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Time_Step'] + list(history_df_rounded.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[history_df_rounded.index] + [history_df_rounded[col] for col in history_df_rounded.columns], fill_color='lavender', align='left'))
    ])
    fig.update_layout(title_text="French-DeGroot Opinion Dynamics History", title_x=0.5)
    fig.write_html("opinion_history_table.html")
    print("Plotly table saved to 'opinion_history_table.html'")

    print(f"\nFinal consensus opinion value (approx): {final_opinions.mean():.4f}")
    print("\nGenerating graph visualization of final opinions...")

    consensus_value = final_opinions.mean()
    consensus_opinions = np.full(g.num_vertices(), consensus_value)
    v_opinions = g.new_vertex_property("double")
    v_opinions.a = consensus_opinions

    cmap = colormaps.get_cmap('plasma')
    color_map = mcolors.LinearSegmentedColormap.from_list("custom_map", cmap.colors)

    gt.graph_draw(g,
                  vertex_fill_color=v_opinions, vertex_color=v_opinions,
                  vcmap=color_map, vertex_size=15, edge_pen_width=1.5,
                  output_size=(800, 800), output="french_degroot_karate_club.png")
    print("Graph saved to 'french_degroot_karate_club.png'")

    # Generate snapshots instead of animation
    create_progression_snapshots(g, history_df)

    # Generate timeseries plots
    create_opinion_timeseries_plots(history_df)

    # Calculate and display social power
    calculate_and_display_social_power(trust_matrix)

def main():
    """
    Main function to run the simulation and generate visualizations.
    """

    stubbornness_coefficient = 0.5
    print("--- Starting French-DeGroot Opinion Dynamics Simulation ---")
    g = gt.collection.data["karate"].copy()
    
    # --- Define a custom initial opinion vector ---
    # Set two leaders (nodes 0 and 33) to opposite opinions (0 and 1)
    # and all others to a neutral 0.5.
    num_nodes = g.num_vertices()
    initial_opinions = np.full(num_nodes, 0.5)
    initial_opinions[0] = 1.0  # Leader 1
    initial_opinions[1] = 0.1  # Leader 2
    initial_opinions[2] = 0.2  # Leader 2
    initial_opinions[33] = 0.0 # Leader 2
    print("\n--- Predicted Consensus Opinion ---")
    print(f"Consensus opinion value (approx): {initial_opinions.mean():.4f}")
    print("\n-------------------------")

    history_df, final_opinions, adj_matrix, influence_matrix, trust_matrix = run_french_degroot_dynamics(g, initial_opinions,stubbornness=stubbornness_coefficient)
    
    print("\n--- Creating Visualizations ---")
    create_visualizations(g, history_df, final_opinions, trust_matrix)

    # --- Create Matrix Heatmaps ---
    print("\n--- Creating Matrix Heatmaps ---")
    create_matrix_heatmap(adj_matrix, "Adjacency Matrix", "adjacency_matrix_heatmap.html")
    create_matrix_heatmap(influence_matrix, "Influence Matrix", "influence_matrix_heatmap.html")
    create_matrix_heatmap(trust_matrix, "Trust Matrix", "trust_matrix_heatmap.html")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()