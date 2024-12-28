import osmnx as ox
import matplotlib.pyplot as plt

# Set the place name and street name
place_name = "Bremen, Germany"
street_name = "Achterdiek"

# Get the graph of the city
G = ox.graph_from_place(place_name, network_type='drive')

# Plot the city map
fig, ax = ox.plot_graph(G, show=False, close=False,node_size=0,edge_color='white',edge_linewidth=0.5)

# Find the location of the street
streets = ox.features_from_place(place_name, {'highway': True})
street = streets[streets['name'] == street_name]

if not street.empty:
    # Plot the street
    street.plot(ax=ax, color=(0.1, 0.2, 0.5, 0.3), linewidth=1.3, markersize=0)
else:
    print(f"Street {street_name} not found in {place_name}")

# Show the plot
# plt.savefig('StreetMap.pdf')
plt.show()
