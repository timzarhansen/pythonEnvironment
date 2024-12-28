import numpy as np
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt


def read_csv(file_path):
    data =  pd.read_csv(file_path, sep=';', header=0)
    return data

resultingListOfOutput = read_csv("tmp/resultsElectionStreet.csv")
resultingListOfOutput = resultingListOfOutput.replace({np.nan: None})

place_name = "Bremen, Germany"
G = ox.graph_from_place(place_name, network_type='walk')

# Plot the city map
fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color='white', edge_linewidth=0.1)

# Find the location of the street
streets = ox.features_from_place(place_name, {'highway': True})


streetIndex = 0
while streetIndex<resultingListOfOutput.shape[0]:
# while streetIndex < 1000:


    street_name = resultingListOfOutput.loc[streetIndex,"Straße"]
    if street_name is None:
        print("The street name is: ",streetIndex)
        streetIndex=streetIndex+1
        continue
    result19 = resultingListOfOutput.loc[streetIndex,"19"]
    result23 = resultingListOfOutput.loc[streetIndex, "23"]
    # colorGreen = (result19-result23)*5 + 0.2
    colorGreen = +0.5-((result19-result23)-0.055)*20
    colorRed = ((result19-result23)-0.055)*20 + 0.5
    if colorRed < 0:
        colorRed = 0.01
    if colorRed > 1:
        colorRed = 0.99
    if colorGreen < 0:
        colorGreen = 0.01
    if colorGreen > 1:
        colorGreen = 0.99

    street = streets[streets['name'] == street_name]
    if not street.empty:
        # Plot the street
        try:
            street.plot(ax=ax, color=[colorRed, colorGreen, 0.0, 0.5], linewidth=0.1, markersize=0)
        except Exception as error:
            print("An exception occurred:", error)
    else:
        print(f"Street {street_name} not found in {place_name}")
    # plt.show()
    # plt.pause(0.05)
    # Show the plot
    streetIndex = streetIndex + 1


plt.savefig('StreetMap.pdf')


