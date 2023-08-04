import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Lists to store the x, y, z, and E values
x_values = []
y_values = []
z_values = []
E_values = []

path = '/usr/workspace/sinha4/GSGM_new/GSGM_for_EIC_Calo/scripts/gen_data/cells.txt'

# Read data from the file
with open(path, 'r') as file:
    for line in file:
        # Split the line into x, y, z, and E values
        x, y, z, E = map(str, line.strip().split())

        input_string = x
        cleaned_string = input_string[1:] #input_string.replace('[', '').strip()
        x = float(cleaned_string)

        y = float(y)
        z = float(z)

        input_string = E
        cleaned_string = input_string[:-1]
        E = float(cleaned_string)    

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        E_values.append(E)

# Create the 3D plot
fig = plt.figure()
fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(111, projection='3d')

# Use a colormap for coloring the points based on E values
# You can choose any colormap from: https://matplotlib.org/stable/tutorials/colors/colormaps.html
colormap = plt.get_cmap('viridis')

# Scatter plot with color based on E values
sc = ax.scatter(x_values, y_values, z_values, c=E_values, cmap=colormap)

# Add a color bar to show the E values corresponding to colors
cbar = plt.colorbar(sc)
cbar.set_label('E values')

# Label the axes and title the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Plot with Color Map of E values')

# Display the plot

plt.show()
