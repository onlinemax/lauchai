#relationship between different features


import matplotlib.pyplot as plt
import pandas as pd

# Assuming your dataframes are named inflation_df and unemployment_df
# And the actual values are in the second column (index 1)

x=df_INFLATION
y=df_UNEMPLOYMENT

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(x.iloc[:, 1], y.iloc[:, 1], alpha=0.7)


# Add labels and title
plt.xlabel(f'Inf', fontsize=12)
plt.ylabel(f'Unp', fontsize=12)
plt.title('Relationship Between Inflation and Unemployment (2013-2024)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()




# ------------------------------------------------------------------------------

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(df_INTEREST.iloc[:, 1], df_UNEMPLOYMENT.iloc[:, 1], alpha=0.7)

# Add labels and title
plt.xlabel('df_INTEREST', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.title('Relationship Between Interest and Unemployment (2013-2024)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(df_INTEREST.iloc[:, 1], df_GDP_REAL.iloc[:, 1], alpha=0.7)

# Add labels and title
plt.xlabel('Interest Rate (%)', fontsize=12)
plt.ylabel('GDP Growth real (%)', fontsize=12)
plt.title('Relationship Between Interest and GDP (2013-2024)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()



# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------


x=df_INFLATION
y=df_GDP_REAL

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(x.iloc[:, 1], y.iloc[:, 1], alpha=0.7)

# Add labels and title
plt.xlabel(f'Inf', fontsize=12)
plt.ylabel(f'GDP', fontsize=12)
plt.title('Relationship Between Inflation and GDP real (2013-2024)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------


x=df_INFLATION
y=df_GDP_PER_CAPITA_REAL

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create the scatter plot
plt.scatter(x.iloc[:, 1], y.iloc[:, 1], alpha=0.7)

# Add labels and title
plt.xlabel(f'Inf', fontsize=12)
plt.ylabel(f'GDP capita', fontsize=12)
plt.title('Relationship Between Inflation and GDP capita (2013-2024)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------