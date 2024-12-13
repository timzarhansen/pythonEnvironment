import osmnx as ox
import matplotlib.pyplot as plt

# Koordinaten für Hamburg
hamburg_center = (53.5510846, 10.0021795)

# Lade die Straßennetzdaten um Hamburg herum
G = ox.graph_from_place('Hamburg, Germany', network_type='drive')

# Plotte die Karte von Hamburg
fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='#777777', edge_linewidth=0.5)

# Suche nach der Straße "Jungfernstieg"
jungfernstieg_edges = ox.graph_from_address('Jungfernstieg, Hamburg, Germany', dist=500, network_type='drive')

# Markiere die Straße "Jungfernstieg" in Rot
ox.plot_graph(jungfernstieg_edges, ax=ax, node_size=0, edge_color='red', edge_linewidth=2)

plt.savefig('StreetMap.pdf')
plt.show()
