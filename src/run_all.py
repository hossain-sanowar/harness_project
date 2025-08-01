from config import (
    OUTPUT_EXCEL_PATH,
    CLUSTER_SIZES,
    STEP_SIZE,
    HPC_WIRE_WEIGHT,
    CENTROID_WIRE_WEIGHT,
    BATTERY_POSITION,
    BATTERY_WIRE_WEIGHT,
    HPC_BATTERY_WIRE_WEIGHT
)

# 🔧 Load shared config
from config import (
    EXCEL_FILE_PATH,
    OUTPUT_EXCEL_PATH,
    CLUSTER_SIZES,
    STEP_SIZE,
    HPC_WIRE_WEIGHT,
    CENTROID_WIRE_WEIGHT,
    BATTERY_POSITION,
    BATTERY_WIRE_WEIGHT,
    HPC_BATTERY_WIRE_WEIGHT
)

# ======================================== 1. start data load ===============================================================

import pandas as pd
import os


 # Define the folder to save plots
data_folder = './data/extract_data/'
os.makedirs(data_folder, exist_ok=True)  # Create the folder if it doesn't exist
 
# Load the Excel file
file_path = EXCEL_FILE_PATH
xls = pd.ExcelFile(file_path)

# --- Extract components_data ---
components_pins = pd.read_excel(file_path, sheet_name='Components Pins List', skiprows=9)
wires_characteristics = pd.read_excel(file_path, sheet_name='Wires Characteristics', skiprows=9)

# Selecting required columns by index
components_data = pd.DataFrame({
    'Index': components_pins.iloc[:, 1],
    'Description': components_pins.iloc[:, 2],
    'Coordinate': components_pins.iloc[:, 17],
    'PinoutNr': components_pins.iloc[:, 24],
    'PinoutSignal': components_pins.iloc[:, 25],
    'PowerSupply': components_pins.iloc[:, 26],
    'Signal': components_pins.iloc[:, 27],
    'Wire_Gauge_[mm²]@85°C_@12V': components_pins.iloc[:, 33],
    'Nominal_Power[W]': components_pins.iloc[:, 18],
    'Nominal_Current_@12V [A]': components_pins.iloc[:, 19],
    'Low_Application': components_pins.iloc[:, 36],
    'Mid_Application': components_pins.iloc[:, 37],
    'High_Application': components_pins.iloc[:, 38],
    'SummerSunnyDayCityIdleTraffic': components_pins.iloc[:, 39],
    'WinterSunnyDayHighwayTraffic': components_pins.iloc[:, 40],
    'WinterRainyNightCityIdleTraffic': components_pins.iloc[:, 41],
    'SummerRainyNightHighwayTraffic': components_pins.iloc[:, 42],

})

# Matching Wire Gauge to Wire Weight
wire_mapping = wires_characteristics.iloc[:, [0, 14]].dropna()
wire_mapping_dict = dict(zip(wire_mapping.iloc[:,0], wire_mapping.iloc[:,1]))
components_data['wire_length_[g/m]'] = components_data['Wire_Gauge_[mm²]@85°C_@12V'].map(wire_mapping_dict)

# --- Extract restricted_data ---
# restricted_coordinates = pd.read_excel(file_path, sheet_name='Restricted Coordinates', skiprows=1, usecols=[0])
restricted_coordinates = pd.read_excel(file_path, sheet_name='Restricted Coordinates', usecols=[0])
restricted_coordinates.columns = ['Restricted_Coordinate']

# --- Extract wire_characteristics_data ---
wire_characteristics_data = pd.DataFrame({
    'wire_type': wires_characteristics.iloc[:, 0],
    'Description': wires_characteristics.iloc[:, 1],
    'Standard': wires_characteristics.iloc[:, 2],
    'Battery_Zone_Connection': wires_characteristics.iloc[:, 4],
    'wire_weight': wires_characteristics.iloc[:, 14],
    'CO2_Emission': wires_characteristics.iloc[:, 15],
    'wire_CurrentCarryingCapacity[A]':wires_characteristics.iloc[:,13]
})

# # --- Extract DrivingScenarios_data ---
# driving_scenarios_data = pd.DataFrame({
#     'Index': components_pins.iloc[:, 1],
#     'PinoutNr': components_pins.iloc[:, 24],
#     'PinoutSignal': components_pins.iloc[:, 25],
#     'PowerSupply': components_pins.iloc[:, 26],
#     'Signal': components_pins.iloc[:, 27],
#     'Nominal_Power[W]': components_pins.iloc[:, 18],
#     'Nominal_Current_@12V [A]': components_pins.iloc[:, 19],
#     'SummerSunnyDayCityIdleTraffic': components_pins.iloc[:, 39],
#     'WinterSunnyDayHighwayTraffic': components_pins.iloc[:, 40],
#     'WinterRainyNightCityIdleTraffic': components_pins.iloc[:, 41],
#     'SummerRainyNightHighwayTraffic': components_pins.iloc[:, 42]
# })


# --- Extract DrivingScenarios_Filterdata ---
components_data = components_data[~((components_data['PowerSupply'] == 'No') & (components_data['Signal'] == 'No'))]
# Remove duplicates based on ['Index', 'PinoutNr']
components_data = components_data.drop_duplicates(subset=['Index', 'PinoutNr'])

# --- Calculate new columns ---
components_data['SummerSunnyDayCityIdleCurrent'] = components_data['Nominal_Current_@12V [A]'] * components_data['SummerSunnyDayCityIdleTraffic']
components_data['WinterSunnyDayHighwayCurrent'] = components_data['Nominal_Current_@12V [A]'] * components_data['WinterSunnyDayHighwayTraffic']
components_data['WinterRainyNightCityIdleCurrent'] = components_data['Nominal_Current_@12V [A]'] * components_data['WinterRainyNightCityIdleTraffic']
components_data['SummerRainyNightHighwayCurrent'] = components_data['Nominal_Current_@12V [A]'] * components_data['SummerRainyNightHighwayTraffic']

# --- Write all to a new Excel file ---
with pd.ExcelWriter(OUTPUT_EXCEL_PATH) as writer:
    components_data.to_excel(writer, sheet_name='components_data', index=False)
    restricted_coordinates.to_excel(writer, sheet_name='restricted_data', index=False)
    wire_characteristics_data.to_excel(writer, sheet_name='wire_characteristics_data', index=False)
    #driving_scenarios_data.to_excel(writer, sheet_name='DrivingScenarios_data', index=False)
    #driving_scenarios_filterdata.to_excel(writer, sheet_name='DrivingScenarios_Filterdata', index=False)
    #driving_scenarios_filterdata.to_excel(writer, sheet_name='DrivingScenarios_FilterCurrentdata', index=False)

print("Data extraction completed and saved to 'extracted_data.xlsx'.")


# ===================================================end data load ========================================================

# ===================================================2. Start Elbow ========================================================

# # 2.0 Elbow Method

# In[52]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path, sheet_name, application_type):
    """Load and preprocess data."""
    # Step 1: Load the data
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Step 2: Filter rows based on application_type
    if application_type in df.columns:
        df = df[df[application_type].astype(str).str.lower() == 'yes']
    else:
        raise ValueError(f"{application_type} column missing in dataset.")

    # Step 3: Ensure required columns are present
    required_columns = ['Coordinate', 'wire_length_[g/m]', 'Index', 'PinoutNr', 'PowerSupply', 'Signal']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Step 4: Convert wire_length_[g/m] to numeric
    df['wire_length_[g/m]'] = pd.to_numeric(
        df['wire_length_[g/m]'].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df[df['wire_length_[g/m]'] > 0]  # Remove invalid or zero values

    # Step 5: Parse and validate coordinates
    df['Coordinate'] = df['Coordinate'].astype(str)
    valid_coordinates = df['Coordinate'].str.contains(r'^-?\d+(\.\d+)?;-?\d+(\.\d+)?$', na=False)
    df = df[valid_coordinates]
    df[['x', 'y']] = df['Coordinate'].str.split(';', expand=True).astype(float)

    # Step 6: Remove rows where both PowerSupply == 'NO' and Signal == 'NO'
    df = df[~((df['PowerSupply'].str.upper() == 'NO') & (df['Signal'].str.upper() == 'NO'))]

    # Step 7: Remove duplicates and NaN values
    df = df.drop_duplicates(subset=['Index', 'PinoutNr'])
    df = df.dropna(subset=['x', 'y', 'wire_length_[g/m]', 'Index', 'PinoutNr'])

    print(f"✅ Preprocessed dataset for {application_type}: {len(df)} rows")
    return df


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_plot_for_all_applications(file_path, sheet_name, application_types):
    """
    Generate a single Elbow plot for all application types with different colors and a legend.
    Parameters:
    - file_path: Path to the input Excel file.
    - sheet_name: Name of the sheet to load data from.
    - application_types: List of application type columns to process (e.g., ['Low_Application', 'Mid_Application', 'High_Application']).
    """
    # Define the folder to save plots
    save_folder = './data/save_plot/elbowPlot'
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    plt.figure(figsize=(10, 6))
    
    # Define colors for each application type
    colors = ['blue', 'orange', 'green']
    
    for app_type, color in zip(application_types, colors):
        # Preprocess the data for the current application type
        preprocessed_data = load_and_preprocess_data(file_path, sheet_name, app_type)
        
        # Extract valid coordinates
        coordinates = np.array(preprocessed_data[['x', 'y']])
        
        # Compute inertia for different numbers of clusters
        inertia = []
        cluster_range = range(1, 15)  # Testing cluster sizes 1 to 10
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(coordinates)
            inertia.append(kmeans.inertia_)
        
        # Plot the Elbow curve for this application type
        plt.plot(cluster_range, inertia, marker='o', linestyle='-', color=color, label=app_type.replace('_Application', ''))
    
    # Add plot title, labels, and legend
    plt.title('Elbow Plot for Different Application Types')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.legend(title='Application Type', loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the save folder
    save_path = os.path.join(save_folder, f'elbow_plot_all_applications.png')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    # # Save and show the plot
    # plt.savefig('elbow_plot_all_applications.png')
    # plt.show()

# Example Usage
file_path = './data/extract_data/extracted_data.xlsx'  # Replace with your Excel file path
sheet_name = 'components_data'  # Replace with your sheet name
application_types = ['Low_Application', 'Mid_Application', 'High_Application']  # Define the application types

elbow_plot_for_all_applications(file_path, sheet_name, application_types)


# =========================================================end elbow =======================================================

# ===================================================3. Start Silhouette ========================================================

# # 3.0 Silhouette Method

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path, sheet_name, application_type):
    """Load and preprocess data."""
    # Step 1: Load the data
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Step 2: Filter rows based on application_type
    if application_type in df.columns:
        df = df[df[application_type].astype(str).str.lower() == 'yes']
    else:
        raise ValueError(f"{application_type} column missing in dataset.")

    # Step 3: Ensure required columns are present
    required_columns = ['Coordinate', 'wire_length_[g/m]', 'Index', 'PinoutNr', 'PowerSupply', 'Signal']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Step 4: Convert wire_length_[g/m] to numeric
    df['wire_length_[g/m]'] = pd.to_numeric(
        df['wire_length_[g/m]'].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df[df['wire_length_[g/m]'] > 0]  # Remove invalid or zero values

    # Step 5: Parse and validate coordinates
    df['Coordinate'] = df['Coordinate'].astype(str)
    valid_coordinates = df['Coordinate'].str.contains(r'^-?\d+(\.\d+)?;-?\d+(\.\d+)?$', na=False)
    df = df[valid_coordinates]
    df[['x', 'y']] = df['Coordinate'].str.split(';', expand=True).astype(float)

    # Step 6: Remove rows where both PowerSupply == 'NO' and Signal == 'NO'
    df = df[~((df['PowerSupply'].str.upper() == 'NO') & (df['Signal'].str.upper() == 'NO'))]

    # Step 7: Remove duplicates and NaN values
    df = df.drop_duplicates(subset=['Index', 'PinoutNr'])
    df = df.dropna(subset=['x', 'y', 'wire_length_[g/m]', 'Index', 'PinoutNr'])

    print(f"✅ Preprocessed dataset for {application_type}: {len(df)} rows")
    return df

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def silhouette_plot_with_average_line(file_path, sheet_name, application_types):
    """
    Generate a single Silhouette plot for all application types with an average line in red.
    Parameters:
    - file_path: Path to the input Excel file.
    - sheet_name: Name of the sheet to load data from.
    - application_types: List of application type columns to process (e.g., ['Low_Application', 'Mid_Application', 'High_Application']).
    """
    # Define the folder to save plots
    save_folder = './data/save_plot/silhouettePlot'
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
    plt.figure(figsize=(10, 6))

    
    # Define colors for each application type
    colors = ['blue', 'orange', 'green']
    all_silhouette_scores = []  # Store all silhouette scores for averaging
    
    for app_type, color in zip(application_types, colors):
        # Preprocess the data for the current application type
        preprocessed_data = load_and_preprocess_data(file_path, sheet_name, app_type)
        
        # Extract valid coordinates
        coordinates = np.array(preprocessed_data[['x', 'y']])
        
        # Compute Silhouette scores for different numbers of clusters
        silhouette_scores = []
        cluster_range = range(2, 15)  # Testing cluster sizes from 2 to 15 (Silhouette requires at least 2 clusters)
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(coordinates)
            score = silhouette_score(coordinates, labels)
            silhouette_scores.append(score)
            all_silhouette_scores.append(score)  # Store scores for averaging
            print(f"Application: {app_type}, Clusters: {k}, Silhouette Score: {score:.4f}")
        
        # Plot the Silhouette curve for this application type
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color=color, label=app_type.replace('_Application', ''))
        
        # Identify the optimal number of clusters for this application type
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        plt.scatter(optimal_k, max(silhouette_scores), color=color, s=150, label=f'{app_type.replace("_Application", "")} Optimal k = {optimal_k}')
        print(f"Optimal number of clusters for {app_type}: {optimal_k}")
    
    # Calculate the average Silhouette score
    average_silhouette_score = np.mean(all_silhouette_scores)
    
    # Plot the average line in red
    plt.axhline(y=average_silhouette_score, color='red', linestyle='--', label='Average Silhouette Score')
    
    # Add plot title, labels, and legend
    plt.title('Silhouette Plot for Different Application Types')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.legend(title='Application Type', loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the save folder
    save_path = os.path.join(save_folder, f'silhouette_plot_all_applications.png')
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

# Example Usage
file_path = './data/extract_data/extracted_data.xlsx'  # Replace with your Excel file path
sheet_name = 'components_data'  # Replace with your sheet name
application_types = ['Low_Application', 'Mid_Application', 'High_Application']  # Define the application types

silhouette_plot_with_average_line(file_path, sheet_name, application_types)


# ===================================================end silhouette ========================================================

# ===================================================4. start K-means ========================================================


# # 4.0 K-Means Clustering Algorithm
# - centroid is not avoided the Restricted Zones
# - saved more data like steps, distance(m), index_weight(g), Total Harness weight
# - Apply Manhatan distance


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ---------------- Step 1: Load and Define Restricted Area ----------------

# Load restricted area data
restricted_file_path = './data/extract_data/extracted_data.xlsx'  # Update with the correct path
restricted_df = pd.read_excel(restricted_file_path, sheet_name='restricted_data')

# Function to parse coordinates
def parse_coordinate(coord):
    try:
        return np.array([float(x) for x in coord.split(";")])
    except:
        return np.array([np.nan, np.nan])

# Apply coordinate parsing
restricted_df['Restricted_Coordinate'] = restricted_df['Restricted_Coordinate'].apply(parse_coordinate)

# Drop invalid coordinates
restricted_df = restricted_df.dropna(subset=['Restricted_Coordinate'])
restricted_df = restricted_df[restricted_df['Restricted_Coordinate'].apply(lambda x: not np.isnan(x).any())]

# Store restricted zones in an array
restricted_zones = np.vstack(restricted_df['Restricted_Coordinate'].values)

# ---------------- Step 2: Apply Application-Based Clustering ----------------

class StarTopologyClustering:
    def __init__(self, file_path, sheet_name, n_clusters, output_prefix):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.n_clusters = n_clusters
        self.output_prefix = output_prefix # Prefix for different output files
        self.df = None
        self.kmeans = None
        self.scaler = MinMaxScaler()
        self.index_data = {} # Store index-based weight data
        self.cluster_weights = defaultdict(float)  # Store total weight of each cluster


    def load_and_preprocess_data(self, application_type):
        """Load and preprocess data."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        # Step 1: Filter rows based on application_type
        if application_type in self.df.columns:
            self.df = self.df[self.df[application_type].astype(str).str.lower() == 'yes']
        else:
            raise ValueError(f"{application_type} column missing in dataset.")

        # Step 2: Ensure required columns are present
        required_columns = ['Coordinate', 'wire_length_[g/m]', 'Index', 'PinoutNr', 'PowerSupply', 'Signal']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Step 3: Convert wire_length_[g/m] to numeric
        self.df['wire_length_[g/m]'] = pd.to_numeric(
            self.df['wire_length_[g/m]'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        self.df = self.df[self.df['wire_length_[g/m]'] > 0]  # Remove invalid or zero values
        # print(f"✅ Valid wire_length_[g/m]:\n{self.df['wire_length_[g/m]'].head()}")

        # Step 4: Parse and validate coordinates
        self.df['Coordinate'] = self.df['Coordinate'].astype(str)
        valid_coordinates = self.df['Coordinate'].str.contains(r'^-?\d+(\.\d+)?;-?\d+(\.\d+)?$', na=False)
        self.df = self.df[valid_coordinates]
        self.df[['x', 'y']] = self.df['Coordinate'].str.split(';', expand=True).astype(float)
        
        # Step 5: Remove rows where both PowerSupply == 'NO' and Signal == 'NO'
        # The tilde (~) negates the condition. It means "keep rows where the condition is not true."
        self.df = self.df[~((self.df['PowerSupply'].str.upper() == 'NO') & (self.df['Signal'].str.upper() == 'NO'))]

        # Step 6: Remove duplicates and NaN values
        self.df = self.df.drop_duplicates(subset=['Index', 'PinoutNr'])
        self.df = self.df.dropna(subset=['x', 'y', 'wire_length_[g/m]', 'Index', 'PinoutNr'])

        print(f"✅ Preprocessed dataset for {application_type}: {len(self.df)} rows")

    def cluster_data(self):
        """Perform K-means clustering on x, y coordinates."""
        feature_matrix = self.df[['x', 'y']].values
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        self.df['Cluster'] = self.kmeans.fit_predict(feature_matrix)

    def calculate_distances_and_weights(self):
        """Compute distances and ensure correct total weight calculations."""
        self.index_data = {}
        self.cluster_weights = defaultdict(float)  # Reset cluster weights

        for cluster_id in range(self.n_clusters):
            cluster_points = self.df[self.df['Cluster'] == cluster_id]
            centroid = self.kmeans.cluster_centers_[cluster_id]
            centroid = self.scaler.inverse_transform([centroid])[0]

            total_cluster_weight = 0  # Track total weight per cluster
            index_weight_map = defaultdict(float)  # Store cumulative weight per Index
            cluster_distance = 0  # Track total distance per cluster

            for _, row in cluster_points.iterrows():
                component_index = row['Index']
                component_coord = (row['x'], row['y'])
                wire_length = row['wire_length_[g/m]']

                # Calculate Manhattan metrics
                manhattan_steps = (abs(component_coord[0] - centroid[0]) + 
                                abs(component_coord[1] - centroid[1]))
                distance_m = manhattan_steps * 0.036
                index_weight_g = distance_m * wire_length

                # Accumulate weight per Index
                index_weight_map[component_index] += wire_length

                # Store/update index data
                if component_index not in self.index_data:
                    self.index_data[component_index] = {
                        'manhattan_steps': manhattan_steps,
                        'distance': distance_m,
                        'total_weight': wire_length,
                        'coordinate': component_coord,
                        'cluster_id': cluster_id,
                        'centroid': centroid
                    }
                else:
                    self.index_data[component_index]['total_weight'] = index_weight_map[component_index]

            # Calculate cluster totals
            total_cluster_weight = sum(index_weight_map.values())
            self.cluster_weights[cluster_id] = total_cluster_weight

    def save_to_excel(self):
        """Enhanced saving with new metrics and sheets."""
        result_df = []
        cluster_weight_tracker = defaultdict(float)
        cluster_distance_tracker = defaultdict(float)
        seen_indices = set()

        for index, data in self.index_data.items():
            # Calculate derived values
            index_weight = data['distance'] * data['total_weight']
            manhattan_steps = data['manhattan_steps']
            
            # Build component row
            result_df.append({
                "Index": index,
                "Cluster ID": data['cluster_id'],
                "Centroid Coordinate": f"{data['centroid'][0]:.2f};{data['centroid'][1]:.2f}",
                "Index Coordinate": f"{data['coordinate'][0]:.2f};{data['coordinate'][1]:.2f}",
                "index Wire Weight(g/m)": data['total_weight'],
                "Manhattan steps": manhattan_steps,
                "distance(m)": data['distance'],
                "index weight (g)": index_weight
            })

            # Update cluster totals (unique indices only)
            if index not in seen_indices:
                cluster_id = data['cluster_id']
                cluster_weight_tracker[cluster_id] += index_weight
                cluster_distance_tracker[cluster_id] += data['distance']
                seen_indices.add(index)

        # Create DataFrames
        result_df = pd.DataFrame(result_df)
        
        cluster_weights_df = pd.DataFrame({
            'Cluster ID': list(cluster_weight_tracker.keys()),
            'Total Component Distance (m)': [round(v, 3) for v in cluster_distance_tracker.values()],
            'Total Weight (g)': [round(v, 3) for v in cluster_weight_tracker.values()]
        })
        
        total_weight_df = pd.DataFrame({
            'Total Component Weight(g)': [round(sum(cluster_weight_tracker.values()), 3)]
        })

        total_distance_df = pd.DataFrame({
            'Total Component Distance(m)': [round(sum(cluster_distance_tracker.values()), 3)]
        })

        
        # Step 2: Define the output directory
        output_directory = "./data/routing/kmeans_clustersCentroidsWithRZ"
        os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist

        # Save to Excel
        output_file = os.path.join(output_directory, f"{self.output_prefix}_{self.application_type}_{self.n_clusters}.xlsx")
        with pd.ExcelWriter(output_file) as writer:
            result_df.to_excel(writer, sheet_name="Cluster_Data", index=False)
            cluster_weights_df.to_excel(writer, sheet_name="Cluster Weights", index=False)
            total_weight_df.to_excel(writer, sheet_name="Total Weight", index=False)
            total_distance_df.to_excel(writer, sheet_name="Total Distance", index=False)

        print(f"Results saved to {output_file}")


    def visualize_clusters(self):
        # Define the folder to save plots
        save_folder = './data/save_plot/kmeans_clustersCentroidsWithRZ'
        os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

        """Plot clusters on the restricted area."""
        plt.figure(figsize=(12, 8))
        plt.title(f'{self.n_clusters} Zones Topology - {self.application_type}')
        plt.xlabel('X Coordinate(mm)')
        plt.ylabel('Y Coordinate(mm)')

        # Plot restricted area
        plt.scatter(restricted_zones[:, 0], restricted_zones[:, 1], color='red', marker='x', s=100, label="Restricted Points")

        colormap = plt.get_cmap('tab10', self.n_clusters)
        colors = [colormap(i) for i in range(self.n_clusters)]

        for cluster_id in range(self.n_clusters):
            cluster_points = self.df[self.df['Cluster'] == cluster_id]
            centroid = self.kmeans.cluster_centers_[cluster_id]
            centroid = self.scaler.inverse_transform([centroid])[0]

            # Plot cluster points and centroid
            plt.scatter(cluster_points['x'], cluster_points['y'], 
                        color=colors[cluster_id], label=f'Cluster {cluster_id}', s=50)
            plt.scatter(centroid[0], centroid[1], c='black', marker='X', 
                        s=100, edgecolors='k')

            # Add cluster ID text with white background
            plt.text(centroid[0], centroid[1], str(cluster_id),
                    fontsize=10, weight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            for _, row in cluster_points.iterrows():
                plt.plot([centroid[0], row['x']], [centroid[1], row['y']], c=colors[cluster_id],
                         linestyle='--', linewidth=0.8)

        # plt.legend()
        # plt.show()
        # Save the plot to the save folder
        save_path = os.path.join(save_folder, f'{self.n_clusters} Zones_Topology_plot_{self.application_type}.png')
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        
        print(f"✅ Plot for {n_clusters} clusters saved to {save_path}")

# ---------------- Step 3: Query for a Specific Index: ----------------

    def get_index_details(self, index_value, n_clusters, application_type):
        """
        Retrieve details for a specific Index and total weight for the corresponding Cluster ID,
        including the number of clusters (n_clusters) and application_type.

        Parameters:
            index_value: The Index to query.
            n_clusters: Number of clusters in the current clustering.
            application_type: The application type (e.g., "Low_Application").

        Returns:
            A formatted string containing Index details, cluster total weight, 
            and application type.
        """
        # Ensure the number of clusters is set correctly
        self.n_clusters = n_clusters
        self.application_type = application_type  # Set the application type

        # Ensure cluster_data has been computed
        if not hasattr(self, 'index_data') or not hasattr(self, 'updated_centroids'):
            raise ValueError("Clusters are not yet computed. Run cluster_data() first.")
        
        # Check if the requested Index exists
        if index_value not in self.index_data:
            raise ValueError(f"Index {index_value} not found in the dataset.")
        
        # Retrieve the data for the specified Index
        data = self.index_data[index_value]
        cluster_id = data['cluster_id']
        centroid = self.kmeans.cluster_centers_[cluster_id]
        
        # Calculate the Cluster Total Weight
        cluster_total_weight = self.cluster_weights.get(cluster_id, 0)
        
        # Build the formatted output
        output = f"""
        Index Details (Application Type = {self.application_type}, n_clusters = {self.n_clusters}):
        ----------------------------------------------------------------
        Index                 : {index_value}
        Cluster ID            : {cluster_id}
        Centroid Coordinate   : {centroid[0]:.2f}; {centroid[1]:.2f}
        Index Coordinate      : {data['coordinate'][0]:.2f}; {data['coordinate'][1]:.2f}
        Distance (m)          : {data['distance']:.2f}
        Total Weight (g)      : {data['distance'] * data['total_weight']:.2f}
        Cluster Total Weight (g): {cluster_total_weight:.2f}
        ----------------------------------------------------------------
        """
        
        # Return the formatted string
        return output


    def get_cluster_total_weights(self, n_clusters, application_type):
        """
        Retrieve total weights for all clusters with a specified number of clusters,
        calculate the overall total weight, and display the application type.

        Parameters:
            n_clusters: Number of clusters to query.
            application_type: The application type (e.g., "Low_Application").

        Returns:
            A formatted string with individual cluster weights, overall total weight,
            and application type.
        """
        # Set the number of clusters and application type
        self.n_clusters = n_clusters
        self.application_type = application_type

        if not hasattr(self, 'cluster_weights'):
            raise ValueError("Cluster weights are not available. Ensure clustering has been run.")

        # Validate that cluster IDs are within the range of n_clusters
        valid_cluster_ids = set(range(self.n_clusters))

        # Filter cluster weights to include only valid clusters
        filtered_cluster_weights = {
            cluster_id: weight
            for cluster_id, weight in self.cluster_weights.items() if cluster_id in valid_cluster_ids
        }

        # Calculate the overall total weight
        overall_total_weight = sum(filtered_cluster_weights.values())

        # Build the formatted output
        output = f"""
        Cluster Total Weights (Application Type = {self.application_type}, n_clusters = {self.n_clusters}):
        ----------------------------------------------------------------
        """
        for cluster_id, weight in filtered_cluster_weights.items():
            output += f"Cluster ID {cluster_id:<12}: {weight:,.2f} g\n"
        output += f"----------------------------------------------------------------\n"
        output += f"Overall Total Weight (g)       : {overall_total_weight:,.2f} g\n"
        output += f"----------------------------------------------------------------"

        return output


    def run(self, application_type):
        """Run the clustering process for a given application type."""
        self.application_type = application_type
        self.load_and_preprocess_data(application_type)
        if not self.df.empty:
            self.cluster_data()
            self.calculate_distances_and_weights()
            self.save_to_excel()
            self.visualize_clusters()
            # self.get_index_details()
            # self.get_cluster_total_weights()
            # Now ready to retrieve Index and Cluster details dynamically
            print(f"Clustering process for {self.application_type} completed successfully.\n")
        else:
            print(f"No data available for {application_type}. Skipping clustering.")

# ---------------- Step 3: Run Clustering and Plot on Restricted Area ----------------


file_path = './data/extract_data/extracted_data.xlsx'
sheet_name = 'components_data'
output_prefix = './cluster_results'

# List of cluster counts to iterate over
cluster_sizes = CLUSTER_SIZES

for n_clusters in cluster_sizes:
    clustering = StarTopologyClustering(file_path, sheet_name, n_clusters, output_prefix)
    for application_type in ["Low_Application", "Mid_Application", "High_Application"]:
        clustering.run(application_type)

# ===================================================end K-means  ========================================================

# ===================================================5. start K-means ARZ ========================================================


# # 5.0 K-Means Clustering Algorithm
# - centroid avoided the Restricted Zones
# - saved more data like steps, distance(m), index_weight(g), Total Harness weight
# - Apply Manhatan distance



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pickle


# ---------------- Step 1: Load Restricted Area ----------------
restricted_file_path = './data/extract_data/extracted_data.xlsx'
restricted_df = pd.read_excel(restricted_file_path, sheet_name='restricted_data')

def parse_coordinate(coord):
    try:
        return np.array([float(x) for x in coord.split(";")])
    except:
        return np.array([np.nan, np.nan])

restricted_df['Restricted_Coordinate'] = restricted_df['Restricted_Coordinate'].apply(parse_coordinate)
restricted_df = restricted_df.dropna(subset=['Restricted_Coordinate'])
restricted_zones = np.vstack(restricted_df['Restricted_Coordinate'].values)
restricted_tree = KDTree(restricted_zones)  


# ---------------- Step 2: Clustering ----------------

class StarTopologyClustering:
    def __init__(self, file_path, sheet_name, n_clusters, output_prefix):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.n_clusters = n_clusters
        self.output_prefix = output_prefix
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        self.updated_centroids = {}

        # Initialize scaler to prevent AttributeError
        self.scaler = StandardScaler()

        # Initialize restricted zones (set to empty if not available)
        self.restricted_zones = self.load_restricted_zones()

    def load_restricted_zones(self):
        """Load restricted zones from the dataset or define manually."""
        if 'Restricted_X' in self.df.columns and 'Restricted_Y' in self.df.columns:
            return self.df[['Restricted_X', 'Restricted_Y']].dropna().values
        else:
            print("⚠️ Warning: No restricted zones found in dataset. Defaulting to empty list.")
            return np.array([])  # Return empty array if no data is available


    def load_and_preprocess_data(self, application_type):
        """Load and preprocess data."""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        # Step 1: Filter rows based on application_type
        if application_type in self.df.columns:
            self.df = self.df[self.df[application_type].astype(str).str.lower() == 'yes']
        else:
            raise ValueError(f"{application_type} column missing in dataset.")

        # Step 2: Ensure required columns are present
        required_columns = ['Coordinate', 'wire_length_[g/m]', 'Index', 'PinoutNr', 'PowerSupply', 'Signal']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Step 3: Convert wire_length_[g/m] to numeric
        self.df['wire_length_[g/m]'] = pd.to_numeric(
            self.df['wire_length_[g/m]'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        self.df = self.df[self.df['wire_length_[g/m]'] > 0]  # Remove invalid or zero values
        # print(f"✅ Valid wire_length_[g/m]:\n{self.df['wire_length_[g/m]'].head()}")

        # Step 4: Parse and validate coordinates
        self.df['Coordinate'] = self.df['Coordinate'].astype(str)
        valid_coordinates = self.df['Coordinate'].str.contains(r'^-?\d+(\.\d+)?;-?\d+(\.\d+)?$', na=False)
        self.df = self.df[valid_coordinates]
        self.df[['x', 'y']] = self.df['Coordinate'].str.split(';', expand=True).astype(float)
        
        # Step 5: Remove rows where both PowerSupply == 'NO' and Signal == 'NO'
        # The tilde (~) negates the condition. It means "keep rows where the condition is not true."
        self.df = self.df[~((self.df['PowerSupply'].str.upper() == 'NO') & (self.df['Signal'].str.upper() == 'NO'))]

        # Step 6: Remove duplicates and NaN values
        self.df = self.df.drop_duplicates(subset=['Index', 'PinoutNr'])
        self.df = self.df.dropna(subset=['x', 'y', 'wire_length_[g/m]', 'Index', 'PinoutNr'])

        print(f"✅ Preprocessed dataset for {application_type}: {len(self.df)} rows")


    def cluster_data(self):
        """Perform K-means clustering on the filtered dataset."""
        feature_matrix = self.scaler.fit_transform(self.df[['x', 'y']])
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        self.df['Cluster'] = self.kmeans.fit_predict(feature_matrix)

        # Ensure updated centroids are numeric
        self.updated_centroids = {i: np.array(centroid) for i, centroid in enumerate(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        )}
        print(f"✅ Initial Updated Centroids: {self.updated_centroids}")


    def find_nearest_valid_centroid(self, centroid, safe_distance=1.0):
        """Find the nearest valid centroid that is not inside a restricted zone."""
        
        # ✅ Check if restricted zones exist
        if self.restricted_zones.size == 0:
            print("⚠️ No restricted zones defined. Returning centroid without changes.")
            return centroid  # No need to adjust if no restricted zones exist

        if np.isnan(centroid).any() or np.isinf(centroid).any():
            print(f"❌ ERROR: Invalid centroid detected before KDTree query: {centroid}")
            return np.array([0, 0])  # Fallback to a valid point

        print(f"🔍 Checking centroid {centroid} against restricted zones...")

        restricted_tree = KDTree(self.restricted_zones)
        direction = np.random.randn(2)  # Random direction if needed
        direction /= np.linalg.norm(direction)  # Normalize

        step_size = 0.5  # Smaller step instead of full jump

        # Move gradually until it's safe
        while restricted_tree.query(centroid)[0] < safe_distance:
            print(f"🚨 Centroid {centroid} too close! Adjusting...")
            centroid += direction * step_size  

            if np.isnan(centroid).any() or np.isinf(centroid).any():
                print(f"❌ ERROR: Centroid became invalid while adjusting: {centroid}")
                return np.array([0, 0])  # Fallback to avoid crash

        print(f"✅ Valid centroid found: {centroid}")
        return centroid


    def optimize_centroids(self):
        """Adjust centroids to avoid restricted zones and minimize total weight."""
        for cluster_id in range(self.n_clusters):
            cluster_points = self.df[self.df['Cluster'] == cluster_id][['x', 'y']].values

            if len(cluster_points) == 0:
                print(f"⚠️ Cluster {cluster_id} has no points. Skipping.")
                continue

            # Compute geometric median (or fallback to mean)
            new_centroid = np.median(cluster_points, axis=0)
            if np.any(np.isnan(new_centroid)):
                print(f"⚠️ Cluster {cluster_id} - Centroid is NaN. Reverting to initial K-means centroid.")
                new_centroid = self.scaler.inverse_transform(self.kmeans.cluster_centers_)[cluster_id]

            # Adjust centroid if it is too close to a restricted zone
            distance, _ = restricted_tree.query(new_centroid)
            if distance < 2.0: # initial 2.0
                direction = new_centroid - restricted_zones[restricted_tree.query(new_centroid)[1]]
                norm = np.linalg.norm(direction)

                if norm == 0:
                    print(f"⚠️ Cluster {cluster_id} - Direction vector has zero magnitude. Skipping adjustment.")
                    continue

                direction /= norm
                new_centroid += direction * (2.0 - distance)

            # Clip centroids to within dataset bounds
            new_centroid[0] = np.clip(new_centroid[0], self.df['x'].min(), self.df['x'].max())
            new_centroid[1] = np.clip(new_centroid[1], self.df['y'].min(), self.df['y'].max())

            # Ensure new centroid is valid
            if np.any(np.isnan(new_centroid)):
                print(f"⚠️ Cluster {cluster_id} - Final centroid is NaN. Using initial K-means value.")
                new_centroid = self.scaler.inverse_transform(self.kmeans.cluster_centers_)[cluster_id]

            self.updated_centroids[cluster_id] = new_centroid

        print(f"✅ Updated Centroids (After Optimization): {self.updated_centroids}")




    def calculate_distances_and_weights(self):
        """Compute distances using validated updated centroids and ensure correct total weight calculations."""
        
        """Compute distances using Manhattan method and calculate derived fields."""
        self.df['Manhattan_steps'] = 0.0
        self.df['distance(m)'] = 0.0
        self.df['index weight (g)'] = 0.0
        self.index_data = {}
        # self.cluster_weights = defaultdict(float)  # Reset cluster weights

        # for cluster_id in range(self.n_clusters):
        #     # Check if the cluster has points
        #     cluster_mask = self.df['Cluster'] == cluster_id
        #     if not cluster_mask.any():
        #         print(f"⚠️ Cluster {cluster_id} has no points. Skipping.")
        #         continue

        #     # Validate and retrieve the centroid from updated_centroids
        #     try:
        #         centroid = np.asarray(self.updated_centroids[cluster_id], dtype=float)
        #     except KeyError:
        #         print(f"⚠️ Missing centroid for Cluster {cluster_id}. Skipping.")
        #         continue
        #     except ValueError:
        #         print(f"⚠️ Invalid centroid format for Cluster {cluster_id}: {self.updated_centroids[cluster_id]}")
        #         continue
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.df['Cluster'] == cluster_id
            if not cluster_mask.any():
                continue

            centroid = self.updated_centroids.get(cluster_id)
            if centroid is None:
                continue

            # Calculate Manhattan distance
            cluster_points = self.df.loc[cluster_mask, ['x', 'y']].values
            delta_x = cluster_points[:, 0] - centroid[0]
            delta_y = cluster_points[:, 1] - centroid[1]
            manhattan_steps = np.abs(delta_x) + np.abs(delta_y)
            
            # Calculate derived values
            self.df.loc[cluster_mask, 'Manhattan_steps'] = manhattan_steps
            self.df.loc[cluster_mask, 'distance(m)'] = manhattan_steps * 0.036
            self.df.loc[cluster_mask, 'index weight (g)'] = (
                self.df.loc[cluster_mask, 'distance(m)'] * 
                self.df.loc[cluster_mask, 'wire_length_[g/m]']
            )

            # # Compute distances and initialize weight tracking
            # cluster_points = self.df.loc[cluster_mask, ['x', 'y']].values  # Extract x, y coordinates of the cluster
            # self.df.loc[cluster_mask, 'Distance_to_Centroid'] = np.linalg.norm(cluster_points - centroid, axis=1)

            # total_cluster_weight = 0  # Track total weight for the cluster
            index_weight_map = defaultdict(float)  # Store cumulative weight per Index

            for _, row in self.df.loc[cluster_mask].iterrows():
                component_index = row['Index']
                component_coord = np.array([row['x'], row['y']])
                wire_length = row['wire_length_[g/m]']


                # Accumulate weight for each unique Index
                index_weight_map[component_index] += wire_length

                # Add or update index data
                if component_index not in self.index_data:
                    self.index_data[component_index] = {
                        'index_wire_weight_(g/m)': wire_length,  # Initialize with wire_length
                        'coordinate': component_coord.tolist(),  # Store as a list
                        'cluster_id': cluster_id,
                        'centroid': centroid.tolist()
                    }
                else:
                    self.index_data[component_index]['index_wire_weight_(g/m)'] = index_weight_map[component_index]

        print("✅ Distances & weights updated successfully!")


    def save_to_excel(self):
        """Save clustering results with new metrics and cluster weight summaries."""
        # Step 1: Prepare component-level data
        result_df = []
        cluster_weight_data = defaultdict(lambda: {'Total Distance(m)': 0, 'Total Weight(g)': 0})

        for index, data in self.index_data.items():
            # Calculate Manhattan distance metrics
            centroid = np.array(data['centroid'])
            component_coord = np.array(data['coordinate'])
            
            # Calculate Manhattan distance
            manhattan_steps = np.sum(np.abs(component_coord - centroid))
            distance_m = manhattan_steps * 0.036
            index_weight_g = distance_m * data['index_wire_weight_(g/m)']
            
            # Update cluster weight totals
            cluster_id = data['cluster_id']
            cluster_weight_data[cluster_id]['Total Distance(m)'] += distance_m
            cluster_weight_data[cluster_id]['Total Weight(g)'] += index_weight_g

            # Build component-level data
            result_df.append({
                "Index": index,
                "Cluster ID": cluster_id,
                "Centroid Coordinate": f"{centroid[0]:.2f};{centroid[1]:.2f}",
                "Index Coordinate": f"{component_coord[0]:.2f};{component_coord[1]:.2f}",
                "index Wire Weight(g/m)": data['index_wire_weight_(g/m)'],
                "Manhattan steps": manhattan_steps,
                "distance(m)": distance_m,
                "index weight (g)": index_weight_g
            })

        # Convert to DataFrames
        result_df = pd.DataFrame(result_df)
        cluster_weights_df = pd.DataFrame([
            {
                'Cluster ID': k,
                'Total Component Distance(m)': v['Total Distance(m)'],
                'Total Weight(g)': v['Total Weight(g)']
            } for k, v in cluster_weight_data.items()
        ])

        # Step 2: Save to Excel
        output_directory = "./data/routing/kmeans_clustersAvoidRZ"
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, 
                                f"{self.output_prefix}_{self.application_type}_{self.n_clusters}.xlsx")
        
        with pd.ExcelWriter(output_file) as writer:
            result_df.to_excel(writer, sheet_name="Cluster_Data", index=False)
            cluster_weights_df.to_excel(writer, sheet_name="Cluster Weights", index=False)
            # Create summary sheet
            pd.DataFrame({'Total Component Weight(g)': [cluster_weights_df['Total Weight(g)'].sum()]})\
            .to_excel(writer, sheet_name="Total Weight", index=False)
            pd.DataFrame({'Total Component Distance(m)': [cluster_weights_df['Total Component Distance(m)'].sum()]})\
            .to_excel(writer, sheet_name="Total Distance", index=False)

        print(f"Results saved to {output_file}")


    def visualize_clusters(self):
        """Plot clusters and restricted zones with cluster IDs on centroids."""
        plt.figure(figsize=(12, 9))
        plt.title(f'{self.n_clusters} Zones Topology - {self.application_type}', fontsize=16)
        plt.xlabel('X Coordinate (mm)', fontsize=12)
        plt.ylabel('Y Coordinate (mm)', fontsize=12)
        
        # Plot restricted zones
        plt.scatter(restricted_zones[:, 0], restricted_zones[:, 1], 
                    color='red', marker='x', s=100, label="Restricted Zones")
        
        # Set the colormap
        colors = plt.get_cmap('tab10', self.n_clusters)
        
        # Plot each cluster with annotations
        for cluster_id in range(self.n_clusters):
            cluster_points = self.df[self.df['Cluster'] == cluster_id]
            if cluster_points.empty:
                continue
                
            centroid = self.updated_centroids[cluster_id]
            
            # Plot cluster points
            plt.scatter(cluster_points['x'], cluster_points['y'], 
                        color=colors(cluster_id), s=40, label=f'Cluster {cluster_id}')
            
            # Plot centroid with cluster ID text
            plt.scatter(centroid[0], centroid[1], marker='X', s=200,
                        edgecolor='black', facecolor=colors(cluster_id))
            plt.text(centroid[0], centroid[1], str(cluster_id),
                    fontsize=12, weight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Draw connection lines
            for _, row in cluster_points.iterrows():
                plt.plot([centroid[0], row['x']], [centroid[1], row['y']], 
                        color=colors(cluster_id), linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Configure legend
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Save plot
        save_folder = './data/save_plot/kmeans_clustersAvoidRZPlot'
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, 
                            f'{self.n_clusters}_Zones_Topology_{self.application_type}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Plot saved to {save_path}")



    def run(self, application_type):
        """Run clustering process with enhanced accuracy and save results."""
        self.application_type = application_type
        self.load_and_preprocess_data(application_type)

        if not self.df.empty:
            self.cluster_data()
            self.optimize_centroids()

            # Check for unassigned components
            unassigned_components = self.df[self.df['Cluster'].isna()]
            if not unassigned_components.empty:
                print("⚠️ WARNING: Some components are NOT assigned to clusters!")
                print(unassigned_components[['x', 'y', 'Index']])

            self.calculate_distances_and_weights()

            # # Save results persistently for reuse
            # model_state_path = f"{self.output_prefix}_{application_type}_{self.n_clusters}.pkl"
            # self.save_model_state(model_state_path)

            # Save to Excel and visualize clusters
            self.save_to_excel()
            self.visualize_clusters()

        else:
            print(f"No data available for {application_type}. Skipping clustering.")



# ---------------- Step 3: Run Clustering ----------------
file_path = './data/extract_data/extracted_data.xlsx'
sheet_name = 'components_data'
output_prefix = './cluster_results'

# List of cluster counts to iterate over
cluster_sizes = CLUSTER_SIZES

for n_clusters in cluster_sizes:
    clustering = StarTopologyClustering(file_path, sheet_name, n_clusters, output_prefix)
    for application_type in ["Low_Application", "Mid_Application", "High_Application"]:
        clustering.run(application_type)


# ===================================================end K-Means RZ ========================================================

# ===================================================6. Start Routing ========================================================

# # 6.0 Routing Path
# - similar with 4.0 Routing Path:
# - save data in more details
# 
# To fix the step calculation issue for A* paths, we need to properly calculate the Manhattan distance for A* paths and ensure the step count is accurate. 
# 
# # Key improvements:
# 
# 1. Accurate Step Calculation:
# 
#     - Calculates both Manhattan distance (manhattan_steps) and actual path length (len(path)-1)
# 
#     -  For A* paths, ensures steps aren't less than Manhattan distance
# 
#     - Stores both values for verification
# 
# 2. Path Validation:
# 
#     - Explicitly checks if A* path makes sense compared to direct distance
# 
#     - Uses Manhattan distance as minimum possible steps
# 
# 3. Enhanced Output:
# 
#     - Added 'Manhattan Steps' and 'Actual Steps' columns for debugging
# 
#     - 'Used Steps' shows which value was actually used in calculations
# 
# 4. Example Case Fix:
# 
#     - For your case (start: (71,22), target: (69,2)):
# 
#         - Manhattan distance: |71-69| + |22-2| = 2 + 20 = 22 steps
# 
#         - If A* returns 24 steps, this is valid (due to obstacles)
# 
#         - If A* returns less than 22, we force it to 22 (minimum possible)
# 
# 5. To use this, update your save_to_excel method to include the new columns in the output. The visualization and other functions remain unchanged.
# 
# - This ensures:
# 
#     - Direct paths use Manhattan distance
# 
#     - A* paths use actual path length (if ≥ Manhattan distance)
# 
#     - Impossible step counts are caught and corrected
# 
#     - Full transparency in step calculations


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings

STEP_SIZE = STEP_SIZE  # Distance per grid step in mm

# ---------------------------- Original Grid Handling ----------------------------
def create_dynamic_grid(data, restricted_coords):
    """Create 1mm-resolution grid matching original visualization"""
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)]
    
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")

# ---------------------------- Revised Visualization Function ----------------------------
def visualize_paths(grid, data, paths, offset, centroid_mapping, plot_title="Routing Visualization"):
    """Enhanced visualization with proper figure handling"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones with subtle background
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot paths with deep colors
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, 
                        label=f'Cluster {cluster_id}')
            except (IndexError, KeyError):
                continue

    # 4. Plot components and centroids with high contrast
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components'
        )
    except ValueError:
        pass

    try:
        centroids = [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate']]
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids'
        )
    except ValueError:
        pass

    # 5. Create optimized legend
    handles, labels = [], []
    for color, cluster_id in zip(colors, unique_clusters):
        handles.append(plt.Line2D([0], [0], color=color, lw=4))
        labels.append(f'Cluster {cluster_id}')
    
    # Add component and centroid entries
    handles.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', 
                  markersize=12, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gold',
                  markersize=15, markeredgecolor='black')
    ])
    labels.extend(['Components', 'Centroids'])
    
    # plt.legend(handles, labels, loc='lower right', 
    #          frameon=True, shadow=True,
    #          title="Legend", title_fontsize=12)


    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()

    
    # Move legend outside plot area
    plt.legend(handles, labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside to the right
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right
    
    
    return fig  # Return the figure object

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.cluster_distances = {}
        self.centroid_mapping = {}

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation
    def process_paths(self):
            """Find paths and track actual connected centroids"""
            centroid_coords = {}
            for idx, row in self.data.iterrows():
                try:
                    centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
                except ValueError as e:
                    print(f"Skipping centroid {idx}: {e}")
            
            for index, row in self.data.iterrows():
                try:
                    start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                    valid_centroids = []
                    
                    # Check all centroids for possible connections
                    for centroid_idx, centroid in centroid_coords.items():
                        path = a_star(self.grid, start, centroid)
                        if path:
                            valid_centroids.append((len(path), centroid_idx))
                    
                    if not valid_centroids:
                        self.paths[index] = []
                        self.centroid_mapping[index] = None
                        continue
                    
                    # Find closest valid centroid
                    _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                    goal = centroid_coords[closest_idx]
                    final_path = a_star(self.grid, start, goal)
                    
                    self.paths[index] = final_path
                    self.centroid_mapping[index] = closest_idx  # Store connected centroid index

                except ValueError as e:
                    print(f"Skipping index {index}: {e}")
                    self.paths[index] = []
                    self.centroid_mapping[index] = None

    def calculate_weights(self):
        
        """Updated to track both weights and distances"""
        self.cluster_weights = {}
        self.cluster_distances = {}  # Reset for each calculation
        """Calculate weights with accurate step counting"""
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Get path and calculate steps
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")

                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                self.cluster_distances[cluster_id] = self.cluster_distances.get(cluster_id, 0) + distance_m  

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")


    def save_to_excel(self, filename):
        """Save results with dynamic column handling"""
        components_df = pd.DataFrame(self.component_details)
        
        # Define base column order
        column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)','Start grid','Target grid',
            'Path Type', 'Manhattan Steps','Actual Steps','Used Steps', 'Distance (m)', 'Index Weight (g)'
        ]
        
        # Add error column if present
        if 'Error' in components_df.columns:
            column_order.append('Error')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with pd.ExcelWriter(filename) as writer:
            components_df[column_order].to_excel(
                writer, sheet_name='Components', index=False)
            
            if self.cluster_weights:
                cluster_df = pd.DataFrame({
                    'Cluster ID': self.cluster_weights.keys(),
                    'Total Weight (g)': [round(v, 2) for v in self.cluster_weights.values()],
                    'Total Distance (m)': [round(self.cluster_distances.get(k, 0), 3) for k in self.cluster_weights.keys()]
                })
                cluster_df.to_excel(writer, sheet_name='Cluster Weights', index=False)
            
            # Create total distance sheet
            if self.cluster_distances:
                total_distance = round(sum(self.cluster_distances.values()), 3)
                distance_df = pd.DataFrame({'Total Harness Distance (m)': [total_distance]})
            
            total_weight = round(sum(self.cluster_weights.values()), 2) if self.cluster_weights else 0
            total_df = pd.DataFrame({'Total Weight (g)': [total_weight]})
            total_df.to_excel(writer, sheet_name='Total Weight', index=False)
            distance_df.to_excel(writer, sheet_name='Total Distance', index=False)

    def visualize(self, plot_title="Routing Results"):
        """Generate and return visualization figure"""
        return visualize_paths(self.grid, self.data, self.paths, 
                              self.grid_offset, self.centroid_mapping,
                              plot_title=plot_title)

# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # Process data
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_paths()
                processor.calculate_weights()
                processor.save_to_excel(output_path)
                # fig = processor.visualize(plot_title=base_name)
                fig = processor.visualize(plot_title=base_name)
                
                # Save and close the figure
                # Save and close the figure
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")
                

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {
        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/routing/kmeans_clustersAvoidRZ",
        "routing_folder": "./data/routing/AGrid_routing",
        "plot_folder": "./data/save_plot/AGrid_routing_plots"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")

# ===================================================end routing ========================================================

# ==============================================7. start centroid-to-HPC connections=========================================

## 7.0 centroid-to-HPC connections
# - Randomly selected the HPC position based on centroids position

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings
from scipy.spatial import distance

STEP_SIZE = STEP_SIZE  # Distance per grid step in mm
# Add new constants near other constants
HPC_WIRE_WEIGHT = HPC_WIRE_WEIGHT  # 6g/m for centroid-to-HPC connections; Simpler cars can operate with 100 Mbit/s Ethernet. For this purpose, FLKS9Y 2x0,13 cables are used

# ---------------------------- Original Grid Handling ----------------------------
def create_dynamic_grid(data, restricted_coords):
    """Create 1mm-resolution grid matching original visualization"""
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)]
    
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")

# ---------------------------- Revised Visualization Function ----------------------------
def visualize_paths(grid, data, paths, offset, centroid_mapping, hpc_position=None, centroid_hpc_paths=None, plot_title="Routing Visualization"):
    """Enhanced visualization with proper figure handling"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones with subtle background
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot paths with deep colors
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, 
                        label=f'Cluster {cluster_id}')
            except (IndexError, KeyError):
                continue

    # 4. Plot components and centroids with high contrast
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components'
        )
    except ValueError:
        pass

    try:
        centroids = [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate']]
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids'
        )
    except ValueError:
        pass

    # Add HPC visualization
    if hpc_position and centroid_hpc_paths:
        # Plot HPC connections
        for path in centroid_hpc_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, 'purple', linestyle=':', 
                        linewidth=3, alpha=0.8, label='HPC Connection')
        
        # Plot HPC node
        hpc_x = hpc_position[0] + min_x
        hpc_y = hpc_position[1] + min_y
        plt.scatter(hpc_x, hpc_y, color='red', s=300, marker='*',
                   edgecolor='black', label='HPC')

    # 5. Create optimized legend
    handles, labels = [], []
    for color, cluster_id in zip(colors, unique_clusters):
        handles.append(plt.Line2D([0], [0], color=color, lw=4))
        labels.append(f'Cluster {cluster_id}')
    
    # Add component and centroid entries
    handles.extend([
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', 
                  markersize=12, markeredgecolor='black'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gold',
                  markersize=15, markeredgecolor='black')
    ])
    labels.extend(['Components', 'Centroids'])

    # Update legend
    handles.append(plt.Line2D([0], [0], color='purple', linestyle=':', lw=3))
    handles.append(plt.scatter([], [], color='red', s=300, marker='*', 
                              edgecolor='black'))
    labels.extend(['HPC Connection', 'HPC'])
    
    # plt.legend(handles, labels, loc='lower right', 
    #          frameon=True, shadow=True,
    #          title="Legend", title_fontsize=12)

    plt.legend(handles, labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right

    
    return fig

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.centroid_mapping = {}
        self.hpc_position = None
        self.centroid_hpc_paths = {}

    def calculate_hpc_position(self, centroid_points):
        """Find optimal HPC position using medoid approach"""
        valid_points = [p for p in centroid_points if self.grid[p[0], p[1]] == 0]
        if not valid_points:
            raise ValueError("No valid HPC positions available")
        
        # Change from 'euclidean' to 'cityblock' for Manhattan distance
        distance_matrix = distance.cdist(valid_points, centroid_points, 'cityblock')
        return valid_points[np.argmin(distance_matrix.sum(axis=1))]

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation

    def process_hpc_paths(self):
        """Process paths from centroids to HPC"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(
                    row['Centroid Coordinate'], self.grid, self.grid_offset
                )
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        try:
            self.hpc_position = self.calculate_hpc_position(list(centroid_coords.values()))
            print(f"HPC established at grid position: {self.hpc_position}")
        except ValueError as e:
            print(f"HPC placement failed: {e}")
            return

        # Find paths from centroids to HPC
        for c_idx, c_pos in centroid_coords.items():
            path = a_star(self.grid, c_pos, self.hpc_position)
            self.centroid_hpc_paths[c_idx] = path if path else []



    def calculate_weights(self):
        """Calculate weights while preserving original coordinates"""
        # Initialize weight trackers
        self.component_weights = defaultdict(float)
        self.hpc_weights = defaultdict(float)
        self.component_distances = defaultdict(float)  # New: Track component distances
        self.hpc_distances = defaultdict(float)       # New: Track HPC distances
        self.hpc_details = []  # Initialize HPC details list
        
        # Original component weight calculation
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values directly from dataframe
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                # Get cluster info from ORIGINAL data
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates for calculations only
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Path calculations
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")
                
                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results with ORIGINAL values
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                
                # Update component distances
                self.component_distances[cluster_id] += distance_m
                self.component_weights[cluster_id] += index_weight

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")

        # Calculate HPC
        # start change for HPC code
        # Initialize HPC details list and processed clusters tracker

        processed_clusters = set()
        
        # Calculate HPC connection weights using Manhattan path distances
        for c_idx, path in self.centroid_hpc_paths.items():
            if not path:
                continue
                
            try:
                cluster_id = self.data.loc[c_idx, 'Cluster ID']
                
                # Skip if we've already processed this cluster
                if cluster_id in processed_clusters:
                    continue
                    
                processed_clusters.add(cluster_id)
                
                # Get grid positions
                centroid_grid = path[0]
                hpc_grid = path[-1]

                # Calculate Manhattan distance for verification
                dx = abs(centroid_grid[0] - hpc_grid[0])
                dy = abs(centroid_grid[1] - hpc_grid[1])
                manhattan_steps = dx + dy

                # Determine path type and actual steps
                line_cells = bresenham_line(centroid_grid, hpc_grid)
                clear_path = all(self.grid[x, y] == 0 for (x, y) in line_cells)

                # # Calculate actual path distance using Manhattan distance
                # distance_mm = 0
                # for i in range(1, len(path)):
                #     x1, y1 = path[i-1]
                #     x2, y2 = path[i]
                    
                #     # Manhattan distance between consecutive points
                #     segment_distance = (abs(x2 - x1) + abs(y2 - y1)) * STEP_SIZE
                #     distance_mm += segment_distance

                if clear_path:
                    steps = manhattan_steps
                    path_type = 'Direct'
                else:
                    path_type = 'A*'
                    steps = len(path) - 1
                    # Ensure steps aren't less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps
                    
                # Calculate distance and weight
                distance_m = (steps * STEP_SIZE) / 1000
                hpc_weight = distance_m * HPC_WIRE_WEIGHT
                    

                
                # Store and accumulate weights (ONCE PER CLUSTER)
                self.hpc_weights[cluster_id] = hpc_weight
                self.hpc_distances[cluster_id] = distance_m

                # Add to HPC details with grid information
                self.hpc_details.append({
                    'Cluster ID': cluster_id,
                    'Centroid Coordinate': self.data.loc[c_idx, 'Centroid Coordinate'],
                    'Start grid': f"{centroid_grid[0]};{centroid_grid[1]}",
                    'HPC Position': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Target grid': f"{hpc_grid[0]};{hpc_grid[1]}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path) - 1,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'HPC Weight (g)': round(hpc_weight, 2)
                })
                
            except Exception as e:
                warnings.warn(f"HPC weight error for cluster {cluster_id}: {e}")

    # end of change for hpc


        # Combine weights
        all_clusters = set(self.component_weights.keys()).union(self.hpc_weights.keys())
        self.cluster_weights = {
            cluster: round(self.component_weights.get(cluster, 0) + self.hpc_weights.get(cluster, 0), 2)
            for cluster in all_clusters
        }



    def save_to_excel(self, filename):
        """Save results with dynamic column handling and HPC integration"""
        # Create DataFrames
        components_df = pd.DataFrame(self.component_details)
        hpc_connections_df = pd.DataFrame(self.hpc_details) if hasattr(self, 'hpc_details') else pd.DataFrame()
        
        # Define column orders
        components_column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)',
            'Start grid', 'Target grid', 'Path Type','Manhattan Steps','Actual Steps','Used Steps', 
            'Distance (m)', 'Index Weight (g)'
        ]
        
        # Define column orders - ADD CLUSTER ID TO HPC COLUMNS
        hpc_column_order = [
            'Cluster ID', 'Centroid Coordinate', 'HPC Position','Start grid','Target grid', 'Path Type', 'Manhattan Steps',
            'Actual Steps','Used Steps', 'Distance (m)', 'HPC Weight (g)'
        ]
        
        cluster_weight_columns = [
            'Cluster ID','Total Component Distance (m)', 'Centroid-HPC Distance (m)',
            'Component Weight (g)','Centroid-HPC Connection Weight (g)', 'Total Weight (g)'
        ]

        # Add error column if present
        if 'Error' in components_df.columns:
            components_column_order.append('Error')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with pd.ExcelWriter(filename) as writer:
            # 1. Components Sheet (Original format)
            components_df[components_column_order].to_excel(
                writer, sheet_name='Components', index=False)

            # # 2. HPC Connections Sheet (New)
            # if not hpc_connections_df.empty:
            #     hpc_connections_df[hpc_column_order].to_excel(
            #         writer, sheet_name='HPC Connections', index=False)

            # start change code
            # 2. HPC Connections Sheet (Updated with Cluster ID)
            if hasattr(self, 'hpc_details') and self.hpc_details:
                hpc_connections_df = pd.DataFrame(self.hpc_details)
                hpc_connections_df[hpc_column_order].to_excel(
                    writer, sheet_name='HPC Connections', index=False)
            else:
                hpc_connections_df = pd.DataFrame()

            # end change code

            # 3. Cluster Weights Sheet (Enhanced)
            cluster_data = []
            for cluster in self.cluster_weights:
                cluster_data.append({
                    'Cluster ID': cluster,
                    'Total Component Distance (m)': round(self.component_distances.get(cluster, 0), 4),
                    'Centroid-HPC Distance (m)': round(self.hpc_distances.get(cluster, 0), 4),
                    'Component Weight (g)': round(self.component_weights.get(cluster, 0), 2),
                    'Centroid-HPC Connection Weight (g)': round(self.hpc_weights.get(cluster, 0), 2),
                    'Total Weight (g)': round(self.cluster_weights.get(cluster, 0), 2)
                })
            
            pd.DataFrame(cluster_data)[cluster_weight_columns].to_excel(
                writer, sheet_name='Cluster Weights', index=False)

            # 4. Total Weight Sheet (Maintained format)
            total_weight_df = pd.DataFrame({'Total Weight (g)': [
                    sum(self.component_weights.values()) + 
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_df.to_excel(writer, sheet_name='Total Weight', index=False)

            # # 5. Total Distance Sheet (Maintained format)
            # total_distance = round(sum(self.cluster_weights.values()), 2)+sum(self.hpc_weights.values())
            # pd.DataFrame({'Total Harness Weight (g)': [total_weight]}).to_excel(
            #     writer, sheet_name='Total Weight', index=False)
            # 5. Total Distance Sheet (NEW)
            total_component_distance = round(sum(self.component_distances.values()), 4)
            total_hpc_distance = round(sum(self.hpc_distances.values()), 4)
            total_distance_df = pd.DataFrame({
                #'Total Component Distance (m)': [total_component_distance],
                #'Total Centroid-HPC Distance (m)': [total_hpc_distance],
                'Total Distance (m)': [round(total_component_distance + total_hpc_distance, 4)]
            })

            # with pd.ExcelWriter(filename) as writer:
            #     # Total weight
            #     total_weight.to_excel(writer, sheet_name='Total Weight', index=False)
            
            # Add new Total Distance sheet
            total_distance_df.to_excel(writer, sheet_name='Total Distance', index=False)



    def visualize(self, plot_title="Routing Results"):
        return visualize_paths(
            self.grid, self.data, self.paths, self.grid_offset, 
            self.centroid_mapping, self.hpc_position, self.centroid_hpc_paths,
            plot_title=plot_title
        )


    def process_paths(self):
        """Find paths and track actual connected centroids"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        for index, row in self.data.iterrows():
            try:
                start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                valid_centroids = []
                
                # Check all centroids for possible connections
                for centroid_idx, centroid in centroid_coords.items():
                    path = a_star(self.grid, start, centroid)
                    if path:
                        valid_centroids.append((len(path), centroid_idx))
                
                if not valid_centroids:
                    self.paths[index] = []
                    self.centroid_mapping[index] = None
                    continue
                
                # Find closest valid centroid
                _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                goal = centroid_coords[closest_idx]
                final_path = a_star(self.grid, start, goal)
                
                self.paths[index] = final_path
                self.centroid_mapping[index] = closest_idx  # Store connected centroid index

            except ValueError as e:
                print(f"Skipping index {index}: {e}")
                self.paths[index] = []
                self.centroid_mapping[index] = None
        
        # Add HPC processing
        self.process_hpc_paths()
        
# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                # output_path = os.path.join(routing_folder, f"routing_{file}")
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # Process data
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_paths()
                processor.calculate_weights()
                processor.save_to_excel(output_path)
                # fig = processor.visualize(plot_title=base_name)
                fig = processor.visualize(plot_title=base_name)
                
                # Save and close the figure
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")
                

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {

        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/routing/kmeans_clustersAvoidRZ",
        "routing_folder": "./data/routing/Centroid_HPCrouting",
        "plot_folder": "./data/save_plot/Centroid_HPCrouting_plots"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")


#====================================end centroid to HPC ==============================================================

# ===================================================8.0 Centroid-to-Centroid connection ===========================================


## 8.0 Centroid-to-Centroid connection for more redundancy
# 
# 1. Centroid-to-centroid connections with max 2 links
# 2. Distance/weight calculations for these connections
# 3. New visualization elements (green dashed lines)
# 4. New "Centroid Connections" sheet in Excel output
# 5. Separate weight constant (2.2g/m) for these connections
# 6. The original component-to-centroid and centroid-to-HPC functionality remains unchanged.


# Add to top with other imports
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings
from math import sqrt
from itertools import combinations
from scipy.spatial import distance

# Update the constants near other constants

# FLRY 2x0,13	TWISTED-PAIR 2x FLCuMg02RY 0.13A / T105 :2,04 g/m
# FLRY 2x0,35	TWISTED-PAIR 2x FLRY 0.35-A / T105 : 4,26 g/m

CENTROID_WIRE_WEIGHT = CENTROID_WIRE_WEIGHT  # centroid-to-centroid connections; CAN-FD lines. They can operate up to 8 Mbit/s, typically 2 Mbit/s. standard ISO 11898. FLRY 2x0,13 or FLRY 2x0,35

STEP_SIZE = STEP_SIZE  # Distance per grid step in mm
# Add new constants near other constants
HPC_WIRE_WEIGHT = HPC_WIRE_WEIGHT  # 6g/m for centroid-to-HPC connections; Simpler cars can operate with 100 Mbit/s Ethernet. For this purpose, FLKS9Y 2x0,13 cables are used

# ---------------------------- Original Grid Handling ----------------------------
def create_dynamic_grid(data, restricted_coords):
    """Create 1mm-resolution grid matching original visualization"""
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)]
    
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")


def visualize_paths(grid, data, paths, offset, centroid_mapping, hpc_position=None, centroid_hpc_paths=None, centroid_connections=None, plot_title="Routing Visualization"):
    """Enhanced visualization with proper figure handling"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones with subtle background
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot component-to-centroid paths with cluster colors
    plotted_clusters = set()
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                # Only add label once per cluster
                if cluster_id not in plotted_clusters:
                    label = f'Cluster {cluster_id}'
                    plotted_clusters.add(cluster_id)
                else:
                    label = None
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, label=label)
            except (IndexError, KeyError):
                continue

    # 4. Plot components and centroids
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components', zorder=4
        )
    except ValueError:
        pass

    try:
        centroids = [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate']]
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids', zorder=5
        )
    except ValueError:
        pass

    # 5. Plot HPC connections and node
    if hpc_position and centroid_hpc_paths:
        # Plot HPC connections
        for path in centroid_hpc_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, 'purple', linestyle=':', 
                        linewidth=3, alpha=0.8, label='HPC Connection', zorder=2)
        
        # Plot HPC node
        hpc_x = hpc_position[0] + min_x
        hpc_y = hpc_position[1] + min_y
        plt.scatter(hpc_x, hpc_y, color='red', s=300, marker='*',
                   edgecolor='black', label='HPC', zorder=6)

    # 6. Plot centroid-to-centroid connections
    if centroid_connections:
        for conn in centroid_connections:
            path = conn['path']
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color='lime', linestyle='--', 
                        linewidth=2.5, alpha=0.9, label='Centroid Connection', zorder=3)

    # 7. Create unified legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    # Ensure correct order of legend entries
    legend_order = [
        'Components', 'Centroids', 'HPC',
        'HPC Connection', 'Centroid Connection'
    ]
    ordered_handles = []
    ordered_labels = []
    
    # # Legend for Centroid to Components Connection
    # for handle, label in zip(unique_handles, unique_labels):
    #     if label.startswith('Cluster'):
    #         ordered_handles.append(handle)
    #         ordered_labels.append(label)
    
    # Legend Components, Centroids, HPC, HPC Connection, Centroid Connection
    for entry in legend_order:
        for handle, label in zip(unique_handles, unique_labels):
            if label == entry:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break
    
    plt.legend(ordered_handles, ordered_labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.legend(ordered_handles, ordered_labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right

    
    return fig

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.centroid_mapping = {}
        self.hpc_position = None
        self.centroid_hpc_paths = {}
        self.centroid_connections = [] 

    def process_centroid_connections(self):
        """Improved centroid connection logic with better logging"""
        #print("\n=== Processing Centroid Connections ===")
        
        # Get valid cluster centroids
        unique_clusters = self.data.groupby('Cluster ID').first()
        cluster_centroids = {}
        for cluster_id, row in unique_clusters.iterrows():
            try:
                original_coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
                grid_pos = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
                #print(f"Cluster {cluster_id}: Original {original_coord} -> Grid {grid_pos}")
                cluster_centroids[cluster_id] = grid_pos
            except ValueError as e:
                print(f"Skipping cluster {cluster_id}: {str(e)}")
                continue

        # Generate all possible pairs
        clusters = list(cluster_centroids.keys())
        pairs = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
                pos_a = cluster_centroids[cluster_a]
                pos_b = cluster_centroids[cluster_b]
                distance = sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                pairs.append((distance, cluster_a, cluster_b, pos_a, pos_b))

        # Sort by distance
        pairs.sort(key=lambda x: x[0])
        
        connection_counts = defaultdict(int)
        self.centroid_connections = []
        total_connections = 0
        
        for pair in pairs:
            distance, cluster_a, cluster_b, pos_a, pos_b = pair
            
            # Skip if either cluster has 2 connections
            if connection_counts[cluster_a] >= 2 or connection_counts[cluster_b] >= 2:
                continue
                
            # Find path
            path = a_star(self.grid, pos_a, pos_b)
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                weight = distance_m * CENTROID_WIRE_WEIGHT
                
                self.centroid_connections.append({
                    'source': cluster_a,
                    'dest': cluster_b,
                    'path': path,
                    'steps': steps,
                    'distance': distance_m,
                    'weight': weight
                })
                
                connection_counts[cluster_a] += 1
                connection_counts[cluster_b] += 1
                total_connections += 1
                #print(f"Connected {cluster_a} <-> {cluster_b} (Distance: {distance:.1f} grid units)")
                
            # Continue even if some clusters reach max connections
            if total_connections >= 2 * len(clusters):
                break

        print(f"Total centroid connections: {len(self.centroid_connections)}")
        
    def calculate_hpc_position(self, centroid_points):
        """Find optimal HPC position using medoid approach"""
        valid_points = [p for p in centroid_points if self.grid[p[0], p[1]] == 0]
        if not valid_points:
            raise ValueError("No valid HPC positions available")
        
        # Change from 'euclidean' to 'cityblock' for Manhattan distance
        distance_matrix = distance.cdist(valid_points, centroid_points, 'cityblock')
        return valid_points[np.argmin(distance_matrix.sum(axis=1))]

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation

    def process_hpc_paths(self):
        """Process paths from centroids to HPC"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(
                    row['Centroid Coordinate'], self.grid, self.grid_offset
                )
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        try:
            self.hpc_position = self.calculate_hpc_position(list(centroid_coords.values()))
            print(f"HPC established at grid position: {self.hpc_position}")
        except ValueError as e:
            print(f"HPC placement failed: {e}")
            return

        # Find paths from centroids to HPC
        for c_idx, c_pos in centroid_coords.items():
            path = a_star(self.grid, c_pos, self.hpc_position)
            self.centroid_hpc_paths[c_idx] = path if path else []



    def calculate_weights(self):
        """Calculate weights while preserving original coordinates"""
        # Initialize weight trackers
        self.component_weights = defaultdict(float)
        self.hpc_weights = defaultdict(float)
        self.component_distances = defaultdict(float)  # New: Track component distances
        self.hpc_distances = defaultdict(float)       # New: Track HPC distances
        self.hpc_details = []  # Initialize HPC details list
        
        # Original component weight calculation
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values directly from dataframe
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                # Get cluster info from ORIGINAL data
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates for calculations only
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Path calculations
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")
                
                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results with ORIGINAL values
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                
                # Update component distances
                self.component_distances[cluster_id] += distance_m
                self.component_weights[cluster_id] += index_weight

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")

        # Calculate HPC
        # start change for HPC code
        # Initialize HPC details list and processed clusters tracker

        processed_clusters = set()
        
        # Calculate HPC connection weights using Manhattan path distances
        for c_idx, path in self.centroid_hpc_paths.items():
            if not path:
                continue
                
            try:
                cluster_id = self.data.loc[c_idx, 'Cluster ID']
                
                # Skip if we've already processed this cluster
                if cluster_id in processed_clusters:
                    continue
                    
                processed_clusters.add(cluster_id)
                
                # Get grid positions
                centroid_grid = path[0]
                hpc_grid = path[-1]

                # Calculate Manhattan distance for verification
                dx = abs(centroid_grid[0] - hpc_grid[0])
                dy = abs(centroid_grid[1] - hpc_grid[1])
                manhattan_steps = dx + dy

                # Determine path type and actual steps
                line_cells = bresenham_line(centroid_grid, hpc_grid)
                clear_path = all(self.grid[x, y] == 0 for (x, y) in line_cells)

                # # Calculate actual path distance using Manhattan distance
                # distance_mm = 0
                # for i in range(1, len(path)):
                #     x1, y1 = path[i-1]
                #     x2, y2 = path[i]
                    
                #     # Manhattan distance between consecutive points
                #     segment_distance = (abs(x2 - x1) + abs(y2 - y1)) * STEP_SIZE
                #     distance_mm += segment_distance

                if clear_path:
                    steps = manhattan_steps
                    path_type = 'Direct'
                else:
                    path_type = 'A*'
                    steps = len(path) - 1
                    # Ensure steps aren't less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps
                    
                # Calculate distance and weight
                distance_m = (steps * STEP_SIZE) / 1000
                hpc_weight = distance_m * HPC_WIRE_WEIGHT
                    

                
                # Store and accumulate weights (ONCE PER CLUSTER)
                self.hpc_weights[cluster_id] = hpc_weight
                self.hpc_distances[cluster_id] = distance_m

                # Add to HPC details with grid information
                self.hpc_details.append({
                    'Cluster ID': cluster_id,
                    'Centroid Coordinate': self.data.loc[c_idx, 'Centroid Coordinate'],
                    'Start grid': f"{centroid_grid[0]};{centroid_grid[1]}",
                    'HPC Position': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Target grid': f"{hpc_grid[0]};{hpc_grid[1]}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path) - 1,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'HPC Weight (g)': round(hpc_weight, 2)
                })
                
            except Exception as e:
                warnings.warn(f"HPC weight error for cluster {cluster_id}: {e}")

    # end of change for hpc



        # Combine weights
        all_clusters = set(self.component_weights.keys()).union(self.hpc_weights.keys())
        self.cluster_weights = {
            cluster: round(self.component_weights.get(cluster, 0) + self.hpc_weights.get(cluster, 0), 2)
            for cluster in all_clusters
        }

        # Add centroid connection weights
        self.centroid_connection_total = 0
        for conn in self.centroid_connections:
            self.centroid_connection_total += conn['weight']
            # Add to individual cluster weights
            # self.cluster_weights[conn['source']] += conn['weight']/2
            # self.cluster_weights[conn['dest']] += conn['weight']/2

        print(f"Total centroid connection weight: {self.centroid_connection_total:.2f}g")



    def save_to_excel(self, filename):
        """Save results with dynamic column handling and HPC integration"""
        # Create DataFrames
        components_df = pd.DataFrame(self.component_details)
        hpc_connections_df = pd.DataFrame(self.hpc_details) if hasattr(self, 'hpc_details') else pd.DataFrame()
        
        # Define column orders
        components_column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)',
            'Start grid', 'Target grid', 'Path Type','Manhattan Steps','Actual Steps','Used Steps', 
            'Distance (m)', 'Index Weight (g)'
        ]
        
        # Define column orders - ADD CLUSTER ID TO HPC COLUMNS
        hpc_column_order = [
            'Cluster ID', 'Centroid Coordinate', 'HPC Position','Start grid','Target grid', 'Path Type', 'Manhattan Steps',
            'Actual Steps','Used Steps', 'Distance (m)', 'HPC Weight (g)'
        ]
        
        cluster_weight_columns = [
            'Cluster ID','Total Component Distance (m)', 'Centroid-HPC Distance (m)',
            'Component Weight (g)','Centroid-HPC Connection Weight (g)', 'Total Weight (g)'
        ]

        # Add error column if present
        if 'Error' in components_df.columns:
            components_column_order.append('Error')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with pd.ExcelWriter(filename) as writer:
            # 1. Components Sheet (Original format)
            components_df[components_column_order].to_excel(
                writer, sheet_name='Components', index=False)

            # # 2. HPC Connections Sheet (New)
            # if not hpc_connections_df.empty:
            #     hpc_connections_df[hpc_column_order].to_excel(
            #         writer, sheet_name='HPC Connections', index=False)

            # start change code
            # 2. HPC Connections Sheet (Updated with Cluster ID)
            if hasattr(self, 'hpc_details') and self.hpc_details:
                hpc_connections_df = pd.DataFrame(self.hpc_details)
                hpc_connections_df[hpc_column_order].to_excel(
                    writer, sheet_name='HPC Connections', index=False)
            else:
                hpc_connections_df = pd.DataFrame()

            # end change code

            # 3. Cluster Weights Sheet (Enhanced)
            cluster_data = []
            for cluster in self.cluster_weights:
                cluster_data.append({
                    'Cluster ID': cluster,
                    'Total Component Distance (m)': round(self.component_distances.get(cluster, 0), 4),
                    'Centroid-HPC Distance (m)': round(self.hpc_distances.get(cluster, 0), 4),
                    'Component Weight (g)': round(self.component_weights.get(cluster, 0), 2),
                    'Centroid-HPC Connection Weight (g)': round(self.hpc_weights.get(cluster, 0), 2),
                    'Total Weight (g)': round(self.cluster_weights.get(cluster, 0), 2)
                })
            
            pd.DataFrame(cluster_data)[cluster_weight_columns].to_excel(
                writer, sheet_name='Cluster Weights', index=False)
            
            
            # 4. Centroid Connections Sheet (Enhanced)
            if self.centroid_connections:
                connection_data = []
                
                # Get original coordinates and grid positions for all clusters
                cluster_info = {}
                unique_clusters = self.data.groupby('Cluster ID').first()
                for cluster_id, row in unique_clusters.iterrows():
                    try:
                        original_coord = row['Centroid Coordinate']
                        grid_pos = float_to_grid(original_coord, self.grid, self.grid_offset)
                        cluster_info[cluster_id] = {
                            'original': original_coord,
                            'grid_x': grid_pos[0],
                            'grid_y': grid_pos[1]
                        }
                    except ValueError:
                        continue

                # Process each connection
                for conn in self.centroid_connections:
                    source = conn['source']
                    dest = conn['dest']
                    steps =  conn['steps']
                    
                    # Get grid coordinates
                    src_grid = (cluster_info[source]['grid_x'], cluster_info[source]['grid_y'])
                    dest_grid = (cluster_info[dest]['grid_x'], cluster_info[dest]['grid_y'])
                    
                    # # Calculate grid distance (Euclidean)
                    # grid_distance = sqrt((src_grid[0]-dest_grid[0])**2 + (src_grid[1]-dest_grid[1])**2)
                    
                    # Calculate grid distance (Manhattan)
                    grid_distance = abs(src_grid[0]-dest_grid[0]) + abs(src_grid[1]-dest_grid[1])
                    
                    connection_data.append({
                        'Source Cluster': source,
                        'Destination Cluster': dest,
                        'Source Original Centroid': cluster_info[source]['original'],
                        'Destination Original Centroid': cluster_info[dest]['original'],
                        'Source Grid': f"{src_grid[0]};{src_grid[1]}",
                        'Destination Grid': f"{dest_grid[0]};{dest_grid[1]}",
                        'Connected Type': f"{source} <-> {dest}",
                        'Grid Distance': round(grid_distance, 2),
                        'steps': steps,
                        'Distance (m)': conn['distance'],
                        'Centroid Connections Weight (g)': conn['weight']
                    })

                centroid_conn_df = pd.DataFrame(connection_data)
                centroid_conn_df = centroid_conn_df[[
                    'Source Cluster', 'Destination Cluster',
                    'Source Original Centroid', 'Destination Original Centroid',
                    'Source Grid', 'Destination Grid', 'Connected Type',
                    'Grid Distance', 'steps', 'Distance (m)',
                    'Centroid Connections Weight (g)'
                ]]
                centroid_conn_df.to_excel(writer, sheet_name='Centroid Connections', index=False)

            # 5. Total Weight Sheet (Maintained format)
            # total_component_weight = round(sum(self.cluster_weights.values()), 2)
            # total_weight = total_component_weight + round(self.centroid_connection_total, 2)+sum(self.hpc_weights.values())
            
            total_weight_All_df = pd.DataFrame({
                'Total Component Weight (g)': [sum(self.component_weights.values())],
                'Total Centroid Connections Weight (g)': [round(self.centroid_connection_total, 2)],
                'Total Centroid-HPC Connections Weight (g)': [sum(self.hpc_weights.values())],
                'Total Harness Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_All_df.to_excel(writer, sheet_name='All Weight', index=False)


            # Total weight
            total_weight_df = pd.DataFrame({'Total Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_df.to_excel(writer, sheet_name='Total Weight', index=False)
        
            # 6. Total Distance Sheet
            total_component_dist = round(sum(self.component_distances.values()), 4)
            total_hpc_dist = round(sum(self.hpc_distances.values()), 4)
            total_centroid_conn_dist = round(sum(conn['distance'] for conn in self.centroid_connections), 4)
            total_distance=total_component_dist+total_hpc_dist+total_centroid_conn_dist
            
            total_distance_All_df = pd.DataFrame({
                'Total Component Distance (m)': [total_component_dist],
                'Total Centroid Connections Distance (m)': [total_centroid_conn_dist],
                'Total Centroid-HPC Connections Distance (m)': [total_hpc_dist],
                'Total Harness Distance (m)': [total_distance]
            })
            
            total_distance_All_df.to_excel(writer, sheet_name='All Distance', index=False)

            # total distance
            total_distance_df = pd.DataFrame({
                'Total Distance (m)': [total_distance]
            })

            total_distance_df.to_excel(writer, sheet_name='Total Distance', index=False)


    # Update the visualize method in RoutingProcessor
    def visualize(self, plot_title="Routing Results"):
        return visualize_paths(
            self.grid, self.data, self.paths, self.grid_offset, 
            self.centroid_mapping, self.hpc_position, self.centroid_hpc_paths,
            self.centroid_connections,  # Pass the centroid connections
            plot_title=plot_title
        )


    def process_paths(self):
        """Find paths and track actual connected centroids"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        for index, row in self.data.iterrows():
            try:
                start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                valid_centroids = []
                
                # Check all centroids for possible connections
                for centroid_idx, centroid in centroid_coords.items():
                    path = a_star(self.grid, start, centroid)
                    if path:
                        valid_centroids.append((len(path), centroid_idx))
                
                if not valid_centroids:
                    self.paths[index] = []
                    self.centroid_mapping[index] = None
                    continue
                
                # Find closest valid centroid
                _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                goal = centroid_coords[closest_idx]
                final_path = a_star(self.grid, start, goal)
                
                self.paths[index] = final_path
                self.centroid_mapping[index] = closest_idx  # Store connected centroid index

            except ValueError as e:
                print(f"Skipping index {index}: {e}")
                self.paths[index] = []
                self.centroid_mapping[index] = None
        
        # Add HPC processing
        self.process_hpc_paths()        
        # Add centroid connections processing
        self.process_centroid_connections()
        
# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                # output_path = os.path.join(routing_folder, f"routing_{file}")
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # Process data
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_paths()
                processor.calculate_weights()
                processor.save_to_excel(output_path)
                # fig = processor.visualize(plot_title=base_name)
                fig = processor.visualize(plot_title=base_name)
                

                # Save and close the figure
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")
                

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {

        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/routing/kmeans_clustersAvoidRZ",
        "routing_folder": "./data/routing/centroid_centroidRouting",
        "plot_folder": "./data/save_plot/centroid_centroidRouting_plot"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")

#======================================================end centroid to centroid=============================================

# ===================================================== 9.0 start centroidHpc To Battery FixedWire Routing =====================
## 9.0 Battery Connection; this is Fixed type of wire
# - Centroids to Battery
# - HPC to Battery
# - Battery Position: 98.0, 18.0
# - Battery Connection wire weight: 4.26 g/m; Wire Type FLRY 0,35 ; fixed wire


# Add to top with other imports
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings
from math import sqrt
from itertools import combinations
from scipy.spatial import distance



STEP_SIZE = STEP_SIZE  # Distance per grid step in mm
# Update the constants near other constants
# FLRY 2x0,13: 2,04 g/m or FLRY 2x0,35 : 4,26 g/m
CENTROID_WIRE_WEIGHT = CENTROID_WIRE_WEIGHT  # centroid-to-centroid connections; CAN-FD lines. They can operate up to 8 Mbit/s, typically 2 Mbit/s. standard ISO 11898. FLRY 2x0,13 or FLRY 2x0,35
# Add new constants near other constants
HPC_WIRE_WEIGHT = HPC_WIRE_WEIGHT  # 6g/m for centroid-to-HPC connections; Simpler cars can operate with 100 Mbit/s Ethernet. For this purpose, FLKS9Y 2x0,13 cables are used

# ---------------------------- Add Battery Constants ----------------------------
BATTERY_POSITION = BATTERY_POSITION # Real-world coordinates
BATTERY_WIRE_WEIGHT = BATTERY_WIRE_WEIGHT  # 4,26 g/m for battery connections; FLRY 0,35 : 4,26 g/m

# ---------------------------- Original Grid Handling ----------------------------

def create_dynamic_grid(data, restricted_coords):
    """Create grid including battery position"""
    # Convert battery position to tuple first
    battery_tuple = tuple(map(float, BATTERY_POSITION.split(';')))
    
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)] + \
        [battery_tuple]  # Add battery position as proper tuple
    
    # Rest of the function remains unchanged
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")


def visualize_paths(grid, data, paths, offset, centroid_mapping, hpc_position=None, 
                    centroid_hpc_paths=None, centroid_connections=None, 
                    battery_paths=None, hpc_battery_path=None,  # New parameters
                    plot_title="Routing Visualization"):
    """Enhanced visualization with battery connections"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones (existing)
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters (existing)
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot component-to-centroid paths with cluster colors
    plotted_clusters = set()
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                # Only add label once per cluster
                if cluster_id not in plotted_clusters:
                    label = f'Cluster {cluster_id}'
                    plotted_clusters.add(cluster_id)
                else:
                    label = None
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, label=label)
            except (IndexError, KeyError):
                continue

    # 4. Plot components and centroids
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components', zorder=4
        )
    except ValueError:
        pass

    try:
        centroids = [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate']]
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids', zorder=5
        )
    except ValueError:
        pass

    # 5. Plot HPC connections and node
    if hpc_position and centroid_hpc_paths:
        # Plot HPC connections
        for path in centroid_hpc_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, 'purple', linestyle=':', 
                        linewidth=3, alpha=0.8, label='HPC-Battery', zorder=2)
        
        # Plot HPC node
        hpc_x = hpc_position[0] + min_x
        hpc_y = hpc_position[1] + min_y
        plt.scatter(hpc_x, hpc_y, color='red', s=300, marker='*',
                   edgecolor='black', label='HPC', zorder=6)

    # 6. Plot centroid-to-centroid connections
    if centroid_connections:
        for conn in centroid_connections:
            path = conn['path']
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color='lime', linestyle='--', 
                        linewidth=2.5, alpha=0.9, label='Centroid-Centroid', zorder=3)

    # 7. Plot battery connections with unified legend
    battery_plotted = False
    if battery_paths:
        for path in battery_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                label = 'Centroid-Battery' if not battery_plotted else None
                plt.plot(x_coords, y_coords, 'darkorange', linestyle='-.', 
                        linewidth=2.5, alpha=0.9, label=label, zorder=3)
                if not battery_plotted:
                    battery_plotted = True

    # 8. NEW: Plot HPC-Battery connection
    if hpc_battery_path:
        x_coords = [p[0] + min_x for p in hpc_battery_path]
        y_coords = [p[1] + min_y for p in hpc_battery_path]
        plt.plot(x_coords, y_coords, 'firebrick', linestyle=':', 
                linewidth=3, alpha=0.8, label='HPC-Battery', zorder=4)

    # 9. Plot Battery node
    # Convert battery string to coordinates
    battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
    plt.scatter(battery_x, battery_y, color='lime', s=350, marker='P',
               edgecolor='black', linewidth=2, label='Battery', zorder=7)

    # 10. Create unified legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # 11. Update legend order
    legend_order = [
        'Components', 'Centroids', 'HPC', 'Battery',
        'Centroids-HPC', 'Centroid-Centroid', 
        'Centroid-Battery', 'HPC-Battery'
    ]

    ordered_handles = []
    ordered_labels = []
    
    # # Legend for Centroid to Components Connection
    # for handle, label in zip(unique_handles, unique_labels):
    #     if label.startswith('Cluster'):
    #         ordered_handles.append(handle)
    #         ordered_labels.append(label)
    
    # Legend Components, Centroids, HPC, HPC Connection, Centroid Connection
    for entry in legend_order:
        for handle, label in zip(unique_handles, unique_labels):
            if label == entry:
                ordered_handles.append(handle)
                ordered_labels.append(label)
                break
    
    plt.legend(ordered_handles, ordered_labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right

    
    return fig

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.centroid_mapping = {}
        self.hpc_position = None
        self.centroid_hpc_paths = {}
        self.centroid_connections = [] 
        self.battery_position = None
        self.centroid_battery_paths = {}
        self.hpc_battery_path = None
        self.battery_weights = defaultdict(float)
        self.battery_paths = {}
        # self.hpc_battery_path = None
        # self.battery_weights = defaultdict(float)


    def process_battery_connections(self):
        """Connect centroids and HPC to battery"""
        try:
            battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
            battery_coord = f"{battery_x};{battery_y}"
            # Store both original and grid positions
            self.battery_position = (battery_x, battery_y)  # Original coordinates
            battery_grid = float_to_grid(battery_coord, self.grid, self.grid_offset)
            self.battery_grid_position = battery_grid  # Grid coordinates
        except ValueError as e:
            print(f"Battery placement failed: {e}")
            self.battery_position = None
            self.battery_grid_position = None
            return

        # Connect centroids to battery using grid position
        cluster_centroids = self.data.groupby('Cluster ID').first()['Centroid Coordinate']
        
        self.centroid_battery_paths = {}
        for cluster_id, centroid_coord in cluster_centroids.items():
            try:
                centroid_pos = float_to_grid(centroid_coord, self.grid, self.grid_offset)
                path = a_star(self.grid, centroid_pos, self.battery_grid_position)
                self.centroid_battery_paths[cluster_id] = path
            except Exception as e:
                print(f"Skipping cluster {cluster_id} battery connection: {e}")

        # Connect HPC to battery using grid position
        if self.hpc_position and self.battery_grid_position:
            try:
                self.hpc_battery_path = a_star(self.grid, self.hpc_position, self.battery_grid_position)
            except Exception as e:
                print(f"HPC-battery connection failed: {e}")
    

    def process_centroid_connections(self):
        """Improved centroid connection logic with better logging"""
        #print("\n=== Processing Centroid Connections ===")
        
        # Get valid cluster centroids
        unique_clusters = self.data.groupby('Cluster ID').first()
        cluster_centroids = {}
        for cluster_id, row in unique_clusters.iterrows():
            try:
                original_coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
                grid_pos = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
                #print(f"Cluster {cluster_id}: Original {original_coord} -> Grid {grid_pos}")
                cluster_centroids[cluster_id] = grid_pos
            except ValueError as e:
                print(f"Skipping cluster {cluster_id}: {str(e)}")
                continue

        # Generate all possible pairs
        clusters = list(cluster_centroids.keys())
        pairs = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
                pos_a = cluster_centroids[cluster_a]
                pos_b = cluster_centroids[cluster_b]
                distance = sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                pairs.append((distance, cluster_a, cluster_b, pos_a, pos_b))

        # Sort by distance
        pairs.sort(key=lambda x: x[0])
        
        connection_counts = defaultdict(int)
        self.centroid_connections = []
        total_connections = 0
        
        for pair in pairs:
            distance, cluster_a, cluster_b, pos_a, pos_b = pair
            
            # Skip if either cluster has 2 connections
            if connection_counts[cluster_a] >= 2 or connection_counts[cluster_b] >= 2:
                continue
                
            # Find path
            path = a_star(self.grid, pos_a, pos_b)
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                weight = distance_m * CENTROID_WIRE_WEIGHT
                
                self.centroid_connections.append({
                    'source': cluster_a,
                    'dest': cluster_b,
                    'path': path,
                    'steps': steps,
                    'distance': distance_m,
                    'weight': weight
                })
                
                connection_counts[cluster_a] += 1
                connection_counts[cluster_b] += 1
                total_connections += 1
                #print(f"Connected {cluster_a} <-> {cluster_b} (Distance: {distance:.1f} grid units)")
                
            # Continue even if some clusters reach max connections
            if total_connections >= 2 * len(clusters):
                break

        print(f"Total centroid connections: {len(self.centroid_connections)}")
        
    def calculate_hpc_position(self, centroid_points):
        """Find optimal HPC position using medoid approach"""
        valid_points = [p for p in centroid_points if self.grid[p[0], p[1]] == 0]
        if not valid_points:
            raise ValueError("No valid HPC positions available")
        
        # Change from 'euclidean' to 'cityblock' for Manhattan distance
        distance_matrix = distance.cdist(valid_points, centroid_points, 'cityblock')
        return valid_points[np.argmin(distance_matrix.sum(axis=1))]

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation

    def process_hpc_paths(self):
        """Process paths from centroids to HPC"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(
                    row['Centroid Coordinate'], self.grid, self.grid_offset
                )
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        try:
            self.hpc_position = self.calculate_hpc_position(list(centroid_coords.values()))
            print(f"HPC established at grid position: {self.hpc_position}")
        except ValueError as e:
            print(f"HPC placement failed: {e}")
            return

        # Find paths from centroids to HPC
        for c_idx, c_pos in centroid_coords.items():
            path = a_star(self.grid, c_pos, self.hpc_position)
            self.centroid_hpc_paths[c_idx] = path if path else []

    def calculate_weights(self):
        """Calculate weights while preserving original coordinates"""
        """Updated weight calculation with battery connections"""
        # Initialize weight trackers
        self.component_weights = defaultdict(float)
        self.hpc_weights = defaultdict(float)
        self.component_distances = defaultdict(float)  # New: Track component distances
        self.hpc_distances = defaultdict(float)       # New: Track HPC distances
        self.hpc_details = []  # Initialize HPC details list
        
        # Original component weight calculation
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values directly from dataframe
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                # Get cluster info from ORIGINAL data
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates for calculations only
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Path calculations
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")
                
                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results with ORIGINAL values
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                
                # Update component distances
                self.component_distances[cluster_id] += distance_m
                self.component_weights[cluster_id] += index_weight

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")

        # Calculate HPC
        # start change for HPC code
        # Initialize HPC details list and processed clusters tracker

        processed_clusters = set()
        
        # Calculate HPC connection weights using Manhattan path distances
        for c_idx, path in self.centroid_hpc_paths.items():
            if not path:
                continue
                
            try:
                cluster_id = self.data.loc[c_idx, 'Cluster ID']
                
                # Skip if we've already processed this cluster
                if cluster_id in processed_clusters:
                    continue
                    
                processed_clusters.add(cluster_id)
                
                # Get grid positions
                centroid_grid = path[0]
                hpc_grid = path[-1]

                # Calculate Manhattan distance for verification
                dx = abs(centroid_grid[0] - hpc_grid[0])
                dy = abs(centroid_grid[1] - hpc_grid[1])
                manhattan_steps = dx + dy

                # Determine path type and actual steps
                line_cells = bresenham_line(centroid_grid, hpc_grid)
                clear_path = all(self.grid[x, y] == 0 for (x, y) in line_cells)

                # # Calculate actual path distance using Manhattan distance
                # distance_mm = 0
                # for i in range(1, len(path)):
                #     x1, y1 = path[i-1]
                #     x2, y2 = path[i]
                    
                #     # Manhattan distance between consecutive points
                #     segment_distance = (abs(x2 - x1) + abs(y2 - y1)) * STEP_SIZE
                #     distance_mm += segment_distance

                if clear_path:
                    steps = manhattan_steps
                    path_type = 'Direct'
                else:
                    path_type = 'A*'
                    steps = len(path) - 1
                    # Ensure steps aren't less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps
                    
                # Calculate distance and weight
                distance_m = (steps * STEP_SIZE) / 1000
                hpc_weight = distance_m * HPC_WIRE_WEIGHT
                    

                
                # Store and accumulate weights (ONCE PER CLUSTER)
                self.hpc_weights[cluster_id] = hpc_weight
                self.hpc_distances[cluster_id] = distance_m

                # Add to HPC details with grid information
                self.hpc_details.append({
                    'Cluster ID': cluster_id,
                    'Centroid Coordinate': self.data.loc[c_idx, 'Centroid Coordinate'],
                    'Start grid': f"{centroid_grid[0]};{centroid_grid[1]}",
                    'HPC Position': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Target grid': f"{hpc_grid[0]};{hpc_grid[1]}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path) - 1,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'HPC Weight (g)': round(hpc_weight, 2)
                })
                
            except Exception as e:
                warnings.warn(f"HPC weight error for cluster {cluster_id}: {e}")

    # end of change for hpc



    
        # Combine weights
        all_clusters = set(self.component_weights.keys()).union(self.hpc_weights.keys())
        self.cluster_weights = {
            cluster: round(self.component_weights.get(cluster, 0) + self.hpc_weights.get(cluster, 0), 2)
            for cluster in all_clusters
        }

        # Add centroid connection weights
        self.centroid_connection_total = 0
        for conn in self.centroid_connections:
            self.centroid_connection_total += conn['weight']
            # Add to individual cluster weights
            # self.cluster_weights[conn['source']] += conn['weight']/2
            # self.cluster_weights[conn['dest']] += conn['weight']/2

        print(f"Total centroid connection weight: {self.centroid_connection_total:.2f}g")

        # NEW: Battery connection weights
        # Centroid-battery weights
        for idx, path in self.battery_paths.items():
            if path:
                cluster_id = self.data.loc[idx, 'Cluster ID']
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                self.battery_weights[cluster_id] += distance_m * BATTERY_WIRE_WEIGHT

        # === Fix battery weight calculation ===
        # Centroid-battery weights
        for cluster_id, path in self.centroid_battery_paths.items():
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                self.battery_weights[cluster_id] += distance_m * BATTERY_WIRE_WEIGHT

        # HPC-battery weight
        if self.hpc_battery_path:
            steps = len(self.hpc_battery_path) - 1
            distance_m = (steps * STEP_SIZE) / 1000
            self.battery_weights['HPC'] = distance_m * BATTERY_WIRE_WEIGHT

        # Update total harness weight
        self.total_harness_weight = (
            sum(self.cluster_weights.values()) + 
            self.centroid_connection_total + 
            sum(self.battery_weights.values())  # Now includes all clusters
        )

        # # HPC-battery weight
        # if self.hpc_battery_path:
        #     steps = len(self.hpc_battery_path) - 1
        #     distance_m = (steps * STEP_SIZE) / 1000
        #     self.battery_weights['HPC'] = distance_m * BATTERY_WIRE_WEIGHT

        # # Update total harness weight
        # self.total_harness_weight = (
        #     sum(self.cluster_weights.values()) +
        #     self.centroid_connection_total +
        #     sum(self.battery_weights.values())
        # )



    def save_to_excel(self, filename):
        """Save results with dynamic column handling and HPC integration"""
        # Create DataFrames
        components_df = pd.DataFrame(self.component_details)
        hpc_connections_df = pd.DataFrame(self.hpc_details) if hasattr(self, 'hpc_details') else pd.DataFrame()
        
        # Define column orders
        components_column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)',
            'Start grid', 'Target grid', 'Path Type','Manhattan Steps','Actual Steps','Used Steps', 
            'Distance (m)', 'Index Weight (g)'
        ]
        
        # Define column orders - ADD CLUSTER ID TO HPC COLUMNS
        hpc_column_order = [
            'Cluster ID', 'Centroid Coordinate', 'HPC Position','Start grid','Target grid', 'Path Type', 'Manhattan Steps',
            'Actual Steps','Used Steps', 'Distance (m)', 'HPC Weight (g)'
        ]
        
        cluster_weight_columns = [
            'Cluster ID','Total Component Distance (m)', 'Centroid-HPC Distance (m)',
            'Component Weight (g)','Centroid-HPC Connection Weight (g)', 'Total Weight (g)'
        ]

        # Add error column if present
        if 'Error' in components_df.columns:
            components_column_order.append('Error')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with pd.ExcelWriter(filename) as writer:
            # 1. Components Sheet (Original format)
            components_df[components_column_order].to_excel(
                writer, sheet_name='Components', index=False)

            # # 2. HPC Connections Sheet (New)
            # if not hpc_connections_df.empty:
            #     hpc_connections_df[hpc_column_order].to_excel(
            #         writer, sheet_name='HPC Connections', index=False)

            # start change code
            # 2. HPC Connections Sheet (Updated with Cluster ID)
            if hasattr(self, 'hpc_details') and self.hpc_details:
                hpc_connections_df = pd.DataFrame(self.hpc_details)
                hpc_connections_df[hpc_column_order].to_excel(
                    writer, sheet_name='HPC Connections', index=False)
            else:
                hpc_connections_df = pd.DataFrame()

            # end change code

            # 3. Cluster Weights Sheet (Enhanced)
            cluster_data = []
            for cluster in self.cluster_weights:
                cluster_data.append({
                    'Cluster ID': cluster,
                    'Total Component Distance (m)': round(self.component_distances.get(cluster, 0), 4),
                    'Centroid-HPC Distance (m)': round(self.hpc_distances.get(cluster, 0), 4),
                    'Component Weight (g)': round(self.component_weights.get(cluster, 0), 2),
                    'Centroid-HPC Connection Weight (g)': round(self.hpc_weights.get(cluster, 0), 2),
                    'Total Weight (g)': round(self.cluster_weights.get(cluster, 0), 2)
                })
            
            pd.DataFrame(cluster_data)[cluster_weight_columns].to_excel(
                writer, sheet_name='Cluster Weights', index=False)
            
            
            # 4. Centroid Connections Sheet (Enhanced)
            if self.centroid_connections:
                connection_data = []
                
                # Get original coordinates and grid positions for all clusters
                cluster_info = {}
                unique_clusters = self.data.groupby('Cluster ID').first()
                for cluster_id, row in unique_clusters.iterrows():
                    try:
                        original_coord = row['Centroid Coordinate']
                        grid_pos = float_to_grid(original_coord, self.grid, self.grid_offset)
                        cluster_info[cluster_id] = {
                            'original': original_coord,
                            'grid_x': grid_pos[0],
                            'grid_y': grid_pos[1]
                        }
                    except ValueError:
                        continue

                # Process each connection
                for conn in self.centroid_connections:
                    source = conn['source']
                    dest = conn['dest']
                    steps =  conn['steps']
                    
                    # Get grid coordinates
                    src_grid = (cluster_info[source]['grid_x'], cluster_info[source]['grid_y'])
                    dest_grid = (cluster_info[dest]['grid_x'], cluster_info[dest]['grid_y'])
                    
                    # # Calculate grid distance (Euclidean)
                    # grid_distance = sqrt((src_grid[0]-dest_grid[0])**2 + (src_grid[1]-dest_grid[1])**2)

                    # Calculate grid distance (Manhattan)
                    grid_distance = abs(src_grid[0]-dest_grid[0]) + abs(src_grid[1]-dest_grid[1])
                    
                    connection_data.append({
                        'Source Cluster': source,
                        'Destination Cluster': dest,
                        'Source Original Centroid': cluster_info[source]['original'],
                        'Destination Original Centroid': cluster_info[dest]['original'],
                        'Source Grid': f"{src_grid[0]};{src_grid[1]}",
                        'Destination Grid': f"{dest_grid[0]};{dest_grid[1]}",
                        'Connected Type': f"{source} <-> {dest}",
                        'Grid Distance': round(grid_distance, 2),
                        'steps': steps,
                        'Distance (m)': conn['distance'],
                        'Centroid Connections Weight (g)': conn['weight']
                    })

                centroid_conn_df = pd.DataFrame(connection_data)
                centroid_conn_df = centroid_conn_df[[
                    'Source Cluster', 'Destination Cluster',
                    'Source Original Centroid', 'Destination Original Centroid',
                    'Source Grid', 'Destination Grid', 'Connected Type',
                    'Grid Distance', 'steps', 'Distance (m)',
                    'Centroid Connections Weight (g)'
                ]]
                centroid_conn_df.to_excel(writer, sheet_name='Centroid Connections', index=False)
            
            
            # 5. Battery Connections Sheet for fixed wire type: BATTERY_WIRE_WEIGHT = 4.26 # CAN-FD
        
            battery_data = []
            
            # Get unique clusters from centroid_battery_paths
            processed_clusters = set()
            
            for cluster_id, path in self.centroid_battery_paths.items():
                if path and self.battery_grid_position and cluster_id not in processed_clusters:
                    try:
                        # Get first component's coordinates from cluster
                        cluster_data = self.data[self.data['Cluster ID'] == cluster_id].iloc[0]
                        steps = len(path) - 1
                        distance_m = (steps * STEP_SIZE) / 1000
                        weight = distance_m * BATTERY_WIRE_WEIGHT
                        
                        battery_data.append({
                            'Cluster ID': cluster_id,
                            'Source Coordinate': cluster_data['Centroid Coordinate'],
                            'Destination Coordinate': BATTERY_POSITION,
                            'Source Grid': f"{path[0][0]};{path[0][1]}",
                            'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
                            'Steps': steps,
                            'Distance (m)': round(distance_m, 4),
                            'Battery Weight (g)': round(weight, 2)
                        })
                        processed_clusters.add(cluster_id)
                    except Exception as e:
                        print(f"Error processing cluster {cluster_id}: {e}")
                    # else:
                    #     # Sum values for existing cluster
                    #     cluster_entries[cluster_id]['Steps'] += steps
                    #     cluster_entries[cluster_id]['Distance (m)'] += distance_m
                    #     cluster_entries[cluster_id]['Battery Weight (g)'] += weight

            # Add aggregated cluster entries
            #battery_data.extend(cluster_entries.values())

            # HPC-battery connection (keep separate as special case)
            if self.hpc_battery_path and self.battery_grid_position:
                hpc_steps = len(self.hpc_battery_path) - 1
                hpc_distance = (hpc_steps * STEP_SIZE) / 1000
                battery_data.append({
                    'Cluster ID': 'HPC',
                    'Source Coordinate': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Destination Coordinate': BATTERY_POSITION,
                    'Source Grid': f"{self.hpc_battery_path[0][0]};{self.hpc_battery_path[0][1]}",
                    'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
                    'Steps': hpc_steps,
                    'Distance (m)': hpc_distance,
                    'Battery Weight (g)': self.battery_weights['HPC']
                })

            pd.DataFrame(battery_data).to_excel(writer, sheet_name='Battery Connections', index=False)

            # 6. Total Weight Sheet (Maintained format)
            total_weight_All_df = pd.DataFrame({
                'Total Component Weight (g)': [sum(self.component_weights.values())],
                'Total Centroid Connections Weight (g)': [self.centroid_connection_total],
                'Total Centroid-HPC Connections Weight (g)': [sum(self.hpc_weights.values())],
                'Total Battery Connections Weight (g)': [sum(self.battery_weights.values())],
                'Total Harness Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_All_df.to_excel(writer, sheet_name='All Weight', index=False)

            # Total weight

            total_weight_df = pd.DataFrame({'Total Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_df.to_excel(writer, sheet_name='Total Weight', index=False)

            # 7. Total Distance Sheet (NEW)
            # Calculate component-to-centroid distances
            total_component_dist = round(sum(self.component_distances.values()), 4)
            
            # Calculate centroid-to-HPC distances
            total_hpc_dist = round(sum(self.hpc_distances.values()), 4)
            
            # Calculate centroid-to-centroid distances
            total_centroid_conn_dist = round(sum(conn['distance'] for conn in self.centroid_connections), 4)
            
            # Calculate battery connection distances
            battery_dist = 0
            # Centroid-battery distances
            for cluster_id, path in self.centroid_battery_paths.items():
                if path:
                    steps = len(path) - 1
                    battery_dist += (steps * STEP_SIZE) / 1000
            # HPC-battery distance
            if self.hpc_battery_path:
                steps = len(self.hpc_battery_path) - 1
                battery_dist += (steps * STEP_SIZE) / 1000
            

            total_distance_All_df = pd.DataFrame({
                'Total Component distance (m)': [total_component_dist],
                'Total Centroid Connections distance (m)': [total_centroid_conn_dist],
                'Total Centroid-HPC Connections distance (m)': [total_hpc_dist],
                'Total Battery Connections distance (m)': [battery_dist],
                'Total Harness distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]
            })
            total_distance_All_df.to_excel(writer, sheet_name='All Distance', index=False)

            # total distance
            total_distance_df= pd.DataFrame({'Total Distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]})
            total_distance_df.to_excel(writer, sheet_name='Total Distance', index=False)

    # Update the visualize method in RoutingProcessor
    def visualize(self, plot_title="Routing Results"):
        return visualize_paths(
            self.grid, self.data, self.paths, self.grid_offset, 
            self.centroid_mapping, self.hpc_position, 
            self.centroid_hpc_paths, self.centroid_connections,
            battery_paths=self.centroid_battery_paths,  # Corrected here
            hpc_battery_path=self.hpc_battery_path,
            plot_title=plot_title
        )


    def process_paths(self):
        """Find paths and track actual connected centroids"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        for index, row in self.data.iterrows():
            try:
                start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                valid_centroids = []
                
                # Check all centroids for possible connections
                for centroid_idx, centroid in centroid_coords.items():
                    path = a_star(self.grid, start, centroid)
                    if path:
                        valid_centroids.append((len(path), centroid_idx))
                
                if not valid_centroids:
                    self.paths[index] = []
                    self.centroid_mapping[index] = None
                    continue
                
                # Find closest valid centroid
                _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                goal = centroid_coords[closest_idx]
                final_path = a_star(self.grid, start, goal)
                
                self.paths[index] = final_path
                self.centroid_mapping[index] = closest_idx  # Store connected centroid index

            except ValueError as e:
                print(f"Skipping index {index}: {e}")
                self.paths[index] = []
                self.centroid_mapping[index] = None
        
        # Add HPC processing
        self.process_hpc_paths()        
        # Add centroid connections processing
        self.process_centroid_connections()
        
    def process_all(self):
        """Complete processing pipeline"""
        self.process_paths()
        self.process_centroid_connections()
        self.process_hpc_paths()
        self.process_battery_connections()  # New step
        self.calculate_weights()

# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                # output_path = os.path.join(routing_folder, f"routing_{file}")
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # # Process data
                # processor = RoutingProcessor(restricted_file, data_path)
                # processor.process_paths()
                # processor.calculate_weights()
                # processor.save_to_excel(output_path)
                # # fig = processor.visualize(plot_title=base_name)
                # fig = processor.visualize(plot_title=base_name)

                # Process with new pipeline
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_all()  # Includes battery connections
                processor.save_to_excel(output_path)
                
                # Generate visualization first
                fig = processor.visualize(plot_title=f"{base_name} with Battery Connections")

                # Save and close the figure
                # Then check and save
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {
        # "restricted_file": "./Restricted_data.xlsx",
        # "cluster_folder": "./input_clusters",
        # "routing_folder": "./output_routing",
        # "plot_folder": "./output_routing_plots"

        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/routing/kmeans_clustersAvoidRZ",
        "routing_folder": "./data/routing/centroidHpcToBatteryFixedWireRouting",
        "plot_folder": "./data/save_plot/centroidHpcToBatteryFixedWireRouting_plots"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")


# ===============================end centroidHpcToBatteryFixedWireRouting =================================================

# =================10. start Merge Data on centroidHpcToBatteryFixedWireRouting ========================================================


## 10. Merge Data:
# - 1. based on Routing Clustering Data all complete path like cluster-centroid, centroid-centroid, centroid-hpc, centroidHpc-Battery
# - 2. centroid-centroid, centroid-hpc, centroidHpc-Battery: wire type fixed here
# - 3. Extracted_data: sheet_name: DrivingScenarios_FilterCurrentdata
# - 4. this is only used to get the data cluster wise. this is clean dataset

import os
import pandas as pd
from pathlib import Path

# Paths
input_owncloud_path = './data/extract_data/extracted_data.xlsx'
input_routing_folder = './data/routing/centroidHpcToBatteryFixedWireRouting' # path is already routed to Battery position direction
output_folder = './data/merge_data'

# Create output directory if not exist
os.makedirs(output_folder, exist_ok=True)

# Load DrivingScenarios_FilterCurrentdata once
driving_current_data = pd.read_excel(input_owncloud_path, sheet_name='components_data')

# Only keep necessary columns
driving_current_data = driving_current_data[
    ['Index',
    'PinoutNr',
    'PinoutSignal',
    'PowerSupply',
    'Signal',
    'Wire_Gauge_[mm²]@85°C_@12V',
    'wire_length_[g/m]',
     'Nominal_Power[W]', 
     'Nominal_Current_@12V [A]',
     'SummerSunnyDayCityIdleTraffic',
     'SummerSunnyDayCityIdleCurrent',
     'WinterSunnyDayHighwayTraffic',
     'WinterSunnyDayHighwayCurrent',
     'WinterRainyNightCityIdleTraffic',
     'WinterRainyNightCityIdleCurrent',
     'SummerRainyNightHighwayTraffic',
     'SummerRainyNightHighwayCurrent']
]

# Iterate through each Excel file in output_routing
for filename in os.listdir(input_routing_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(input_routing_folder, filename)
        
        # Load Components sheet
        try:
            components_data = pd.read_excel(file_path, sheet_name='Components')
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue
        
        # Merge on 'Index'
        merged_data = pd.merge(driving_current_data, components_data, on='Index', how='inner')

        # Save merged data to a new file
        # new_filename = f"Merge_DrivingScenarios_{Path(filename).stem}.xlsx"
        new_filename = f"{Path(filename).stem.replace('routing_', '')}.xlsx"
        # new_filename = f"{Path(filename).stem}.xlsx"

        output_path = os.path.join(output_folder, new_filename)
        # output_path = os.path.join(output_folder, filename)


        with pd.ExcelWriter(output_path) as writer:
            merged_data.to_excel(writer, sheet_name='Merge_DrivingScenariosCurrent', index=False)

print("✅ All files processed and saved in 'data/merge_data' folder.")

# ===================================================end merge data ========================================================

# ===================================================10.2 start Generated Wire Type based on nominal power ========================================================


## 10.2 Automatically Generated Wire Type for Centroid to Battery and HPC to Battery

# ### a. Centroids to Battery
# - summ of all nominal power by cluster wise
# - cross check with current carrying capacity from wire_characteristics_data
# - select Wire Type based on based on (CurrentCarryingCapacity>=MaxDemanedCurrent) and Battery_Zone_Connection =Yes 
# - get wire_weight[g/m]

# ### b. HPC to Battery
# - sum of all centroid nominal power
# - cross check with current carrying capacity from wire_characteristics_data
# - select Wire Type based on based on (CurrentCarryingCapacity>=MaxDemanedCurrent) and Battery_Zone_Connection =Yes 
# - get wire_weight[g/m]


import os
import pandas as pd
from pathlib import Path

# Paths
input_owncloud_path = './data/extract_data/extracted_data.xlsx'
input_driving_power_folder = './data/merge_data'
output_folder = './data/GetWireTypeCentroids'

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Load wire characteristics
wire_characteristics = pd.read_excel(
    input_owncloud_path, 
    sheet_name='wire_characteristics_data'
)[['wire_type', 'wire_weight', 'CO2_Emission', 'wire_CurrentCarryingCapacity[A]', 'Battery_Zone_Connection']]

wire_characteristics = wire_characteristics.rename(columns={
    'wire_CurrentCarryingCapacity[A]': 'CurrentCarryingCapacity[A]',
    'wire_weight': 'wire_weight[g/m]'
})

# Iterate through files
for filename in os.listdir(input_driving_power_folder):
    if not filename.endswith('.xlsx'): 
        continue
        
    file_path = os.path.join(input_driving_power_folder, filename)
    
    try:
        # Load data and filter for PowerSupply='Yes'
        driving_power_data = pd.read_excel(file_path, sheet_name='Merge_DrivingScenariosCurrent')
        power_supply_data = driving_power_data[
            (driving_power_data['PowerSupply'].str.upper() == 'YES') &
            (pd.to_numeric(driving_power_data['Nominal_Current_@12V [A]'], errors='coerce') > 0)
        ].copy()
        
        if power_supply_data.empty:
            print(f"⚠️ No power supply components with valid current in {filename}. Skipping.")
            continue

        # Calculate total current per cluster
        grouped = power_supply_data.groupby('Cluster ID').agg({
            'Nominal_Current_@12V [A]': 'sum'
        }).reset_index().rename(columns={
            'Nominal_Current_@12V [A]': 'CentroidCurrent'
        })

        # Find appropriate wire for each cluster
        def find_appropriate_wire(current):
            # Filter battery wires
            battery_wires = wire_characteristics[
                wire_characteristics['Battery_Zone_Connection'] == 'Yes'
            ].copy()
            
            # Find wires that meet current requirement
            sufficient_wires = battery_wires[
                battery_wires['CurrentCarryingCapacity[A]'] >= current
            ]
            
            if not sufficient_wires.empty:
                # Get smallest sufficient wire
                wire = sufficient_wires.nsmallest(1, 'CurrentCarryingCapacity[A]').iloc[0]
                return pd.Series([
                    wire['wire_type'],
                    wire['CurrentCarryingCapacity[A]'],
                    wire['wire_weight[g/m]'],
                    'Sufficient'
                ])
            
            # If no sufficient wire, find closest lower capacity wire
            if not battery_wires.empty:
                # Calculate difference below required current
                battery_wires['capacity_diff'] = current - battery_wires['CurrentCarryingCapacity[A]']
                # Filter for wires below current and get closest
                below_wires = battery_wires[battery_wires['CurrentCarryingCapacity[A]'] < current]
                if not below_wires.empty:
                    wire = below_wires.nsmallest(1, 'capacity_diff').iloc[0]
                    return pd.Series([
                        wire['wire_type'],
                        wire['CurrentCarryingCapacity[A]'],
                        wire['wire_weight[g/m]'],
                        'Insufficient'
                    ])
            
            # Default if no wires found
            return pd.Series(['No Suitable Wire', None, None, 'Error'])

        # Apply wire selection
        result_cols = grouped['CentroidCurrent'].apply(find_appropriate_wire)
        result_cols.columns = ['battery_wire_type', 'CurrentCarryingCapacity[A]', 
                              'battery_wire_weight[g/m]', 'Wire_Status']
        grouped = pd.concat([grouped, result_cols], axis=1)

        # Save results
        new_filename = f"{Path(filename).stem.replace('routing_', '')}.xlsx"
        output_path = os.path.join(output_folder, new_filename)
        
        with pd.ExcelWriter(output_path) as writer:
            grouped.to_excel(writer, sheet_name='CentroidsGetWireType', index=False)
            
        print(f"✅ Processed {filename} → {new_filename}")

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")

print("✅ All files processed. Battery-centroid wires sized with fallback to closest lower capacity when needed.")


# ===================================================end automatically wire data ========================================================

# =================================================== 10.3 merge data centroid-to-battery based on nominal power based wire ========================================================

# # 10.3 Merge data based on Nominal Current@12V for kmeans_clustersBattery
# - Merge k-means and Wire type based on Nominal Current data 10.2 file name
# - Wire type define automatically based on sum of all indexes per cluster
# - to calculate the weight and distance for centroid-to-battery based on nominal power based wire

import os
import pandas as pd

# Define paths
cluster_input_folder = './data/routing/kmeans_clustersAvoidRZ'
wire_input_folder = './data/GetWireTypeCentroids'
output_folder = './data/kmeans_clustersWireTypeCentroidsToBattery'



# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all cluster result files
for filename in os.listdir(cluster_input_folder):
    if filename.endswith('.xls') or filename.endswith('.xlsx'):
        cluster_file_path = os.path.join(cluster_input_folder, filename)
        wire_file_path = os.path.join(wire_input_folder, filename)

        # Check if corresponding wire_typeDynamic file exists
        if not os.path.exists(wire_file_path):
            print(f"⚠️  Skipping {filename}: No matching file in GetWireType.")
            continue

        try:
            # Read sheets
            cluster_df = pd.read_excel(cluster_file_path, sheet_name='Cluster_Data')
            wire_df = pd.read_excel(wire_file_path, sheet_name='CentroidsGetWireType')
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        # Ensure required column exists
        if 'Cluster ID' not in cluster_df.columns or 'Cluster ID' not in wire_df.columns:
            print(f"⚠️  Skipping {filename}: Missing 'Cluster ID' in one of the sheets.")
            continue

        # Select needed columns from wire data
        wire_df_filtered = wire_df[['Cluster ID', 'battery_wire_type', 'battery_wire_weight[g/m]']]

        # Merge on Cluster ID
        merged_df = pd.merge(cluster_df, wire_df_filtered, on='Cluster ID', how='left')

        # Save merged result
        output_path = os.path.join(output_folder, filename)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='Cluster_Data', index=False)

        print(f"✅ Merged and saved: {filename}")

print("🏁 All matched files processed and saved to 'kmeans_clustersWireTypeCentroidsToBattery'.")


# ====================================== 10.5 centroid to Battery without HPC-to-Battery based on nominal power demand ==============================

# 10.5 Battery Connection based on Nominal Power
# - similar as 10.4, but only change HPC to Batter connection; no HPC-Battery
# - Centroids to Battery
# - Battery Position: 98.0, 18.0
# - Battery Connection wire weight: Nominal Power wire type
# - use kmeans_clustersBattery data

# Add to top with other imports
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings
from math import sqrt
from itertools import combinations
from scipy.spatial import distance



STEP_SIZE = STEP_SIZE  # Distance per grid step in mm
# Update the constants near other constants
# FLRY 2x0,13: 2,04 g/m or FLRY 2x0,35 : 4,26 g/m
CENTROID_WIRE_WEIGHT = CENTROID_WIRE_WEIGHT  # centroid-to-centroid connections; CAN-FD lines. They can operate up to 8 Mbit/s, typically 2 Mbit/s. standard ISO 11898. FLRY 2x0,13 or FLRY 2x0,35
# Add new constants near other constants
HPC_WIRE_WEIGHT = HPC_WIRE_WEIGHT  # 6g/m for centroid-to-HPC connections; Simpler cars can operate with 100 Mbit/s Ethernet. For this purpose, FLKS9Y 2x0,13 cables are used

# ---------------------------- Add Battery Constants ----------------------------
BATTERY_POSITION = BATTERY_POSITION # Real-world coordinates
BATTERY_WIRE_WEIGHT = BATTERY_WIRE_WEIGHT  # 4,26 g/m for battery connections; FLRY 0,35 : 4,26 g/m
HPC_BATTERY_WIRE_WEIGHT = HPC_BATTERY_WIRE_WEIGHT  # 6g/m for HPC-battery connections (example value):	FLKS9Y 2x0.13 (Ethernet)

# ---------------------------- Original Grid Handling ----------------------------

def create_dynamic_grid(data, restricted_coords):
    """Create grid including battery position"""
    # Convert battery position to tuple first
    battery_tuple = tuple(map(float, BATTERY_POSITION.split(';')))
    
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)] + \
        [battery_tuple]  # Add battery position as proper tuple
    
    # Rest of the function remains unchanged
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")


def visualize_paths(grid, data, paths, offset, centroid_mapping, hpc_position=None, 
                    centroid_hpc_paths=None, centroid_connections=None, 
                    battery_paths=None, hpc_battery_path=None,  
                    plot_title="Routing Visualization"):
    """Enhanced visualization with battery connections"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot component-to-centroid paths
    plotted_clusters = set()
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                if cluster_id not in plotted_clusters:
                    label = f'Cluster {cluster_id}'
                    plotted_clusters.add(cluster_id)
                else:
                    label = None
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, label=label)
            except (IndexError, KeyError):
                continue

    # 4. Plot components
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components', zorder=4
        )
    except ValueError:
        pass

    # 5. Plot centroids with cluster IDs
    try:
        centroids_df = data.groupby('Cluster ID').first()
        centroids = []
        cluster_ids = []
        for cluster_id, row in centroids_df.iterrows():
            coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
            centroids.append(coord)
            cluster_ids.append(cluster_id)
        
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids', zorder=5
        )
        
        # Add cluster ID text labels
        for (x, y), cluster_id in zip(centroids, cluster_ids):
            plt.text(x, y, str(cluster_id), 
                    fontsize=10, weight='bold', color='black',
                    ha='center', va='center', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    except Exception as e:
        print(f"Error plotting centroids: {e}")

    # 6. Plot HPC connections and node
    if hpc_position and centroid_hpc_paths:
        for path in centroid_hpc_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, 'purple', linestyle=':', 
                        linewidth=3, alpha=0.8, label='Centroid-HPC', zorder=2)
        
        hpc_x = hpc_position[0] + min_x
        hpc_y = hpc_position[1] + min_y
        plt.scatter(hpc_x, hpc_y, color='red', s=300, marker='*',
                   edgecolor='black', label='HPC', zorder=6)

    # 7. Plot centroid-to-centroid connections
    if centroid_connections:
        for conn in centroid_connections:
            path = conn['path']
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color='lime', linestyle='--', 
                        linewidth=2.5, alpha=0.9, label='Centroid Links', zorder=3)

    # 8. Plot battery connections
    battery_plotted = False
    if battery_paths:
        for path in battery_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                label = 'Centroid-Battery' if not battery_plotted else None
                plt.plot(x_coords, y_coords, 'darkorange', linestyle='-.', 
                        linewidth=2.5, alpha=0.9, label=label, zorder=3)
                if not battery_plotted:
                    battery_plotted = True

    # 9. Plot Battery node
    battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
    plt.scatter(battery_x, battery_y, color='lime', s=350, marker='P',
               edgecolor='black', linewidth=2, label='Battery', zorder=7)

    # 10. Create unified legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    
    legend_order = [
        'Components', 'Centroids', 'Centroid Links', 'HPC',
        'Centroid-HPC', 'Battery', 'Centroid-Battery'
    ] + [f'Cluster {cid}' for cid in unique_clusters]
    
    ordered_handles = [unique[l] for l in legend_order if l in unique]
    ordered_labels = [l for l in legend_order if l in unique]
    
    plt.legend(ordered_handles, ordered_labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right

    
    return fig

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
 
        # # New: Store battery wire weights per cluster
        # self.cluster_battery_weights = self.data.groupby('Cluster ID')['battery_wire_weight[g/m]'].first().to_dict()

        # # Debug: Print column names
        # print("\n=== Columns in Data ===")
        # print(self.data.columns.tolist())
        
        # # Debug: Print sample battery weights
        # print("\n=== Battery Weight Samples ===")
        # if 'battery_wire_weight[g/m]' in self.data.columns:
        #     print(self.data[['Cluster ID', 'battery_wire_weight[g/m]']].head(3))
        # else:
        #     print("Column 'battery_wire_weight[g/m]' not found!")
        
        
        # Load weights (now with string IDs)
        self.cluster_battery_weights = self.data.groupby('Cluster ID')['battery_wire_weight[g/m]'].first().to_dict()
        
        # # Debug: Print loaded weights
        # print("\n=== Loaded Battery Weights ===")
        # print(self.cluster_battery_weights)

        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.centroid_mapping = {}
        self.hpc_position = None
        self.centroid_hpc_paths = {}
        self.centroid_connections = [] 
        self.battery_position = None
        self.centroid_battery_paths = {}
        self.hpc_battery_path = None
        self.battery_weights = defaultdict(float)
        self.battery_paths = {}
        self.cluster_weights_centroid ={}
        # self.hpc_battery_path = None
        # self.battery_weights = defaultdict(float)


    def process_battery_connections(self):
        """Connect centroids and HPC to battery"""
        try:
            battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
            battery_coord = f"{battery_x};{battery_y}"
            # Store both original and grid positions
            self.battery_position = (battery_x, battery_y)  # Original coordinates
            battery_grid = float_to_grid(battery_coord, self.grid, self.grid_offset)
            self.battery_grid_position = battery_grid  # Grid coordinates
        except ValueError as e:
            print(f"Battery placement failed: {e}")
            self.battery_position = None
            self.battery_grid_position = None
            return

        # Connect centroids to battery using grid position
        cluster_centroids = self.data.groupby('Cluster ID').first()['Centroid Coordinate']
        
        self.centroid_battery_paths = {}
        for cluster_id, centroid_coord in cluster_centroids.items():
            try:
                centroid_pos = float_to_grid(centroid_coord, self.grid, self.grid_offset)
                path = a_star(self.grid, centroid_pos, self.battery_grid_position)
                self.centroid_battery_paths[cluster_id] = path
            except Exception as e:
                print(f"Skipping cluster {cluster_id} battery connection: {e}")

        # # Connect HPC to battery using grid position
        # if self.hpc_position and self.battery_grid_position:
        #     try:
        #         self.hpc_battery_path = a_star(self.grid, self.hpc_position, self.battery_grid_position)
        #     except Exception as e:
        #         print(f"HPC-battery connection failed: {e}")
    

    def process_centroid_connections(self):
        """Improved centroid connection logic with better logging"""
        #print("\n=== Processing Centroid Connections ===")
        
        # Get valid cluster centroids
        unique_clusters = self.data.groupby('Cluster ID').first()
        cluster_centroids = {}
        for cluster_id, row in unique_clusters.iterrows():
            try:
                original_coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
                grid_pos = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
                #print(f"Cluster {cluster_id}: Original {original_coord} -> Grid {grid_pos}")
                cluster_centroids[cluster_id] = grid_pos
            except ValueError as e:
                print(f"Skipping cluster {cluster_id}: {str(e)}")
                continue

        # Generate all possible pairs
        clusters = list(cluster_centroids.keys())
        pairs = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
                pos_a = cluster_centroids[cluster_a]
                pos_b = cluster_centroids[cluster_b]
                distance = sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                pairs.append((distance, cluster_a, cluster_b, pos_a, pos_b))

        # Sort by distance
        pairs.sort(key=lambda x: x[0])
        
        connection_counts = defaultdict(int)
        self.centroid_connections = []
        total_connections = 0
        
        for pair in pairs:
            distance, cluster_a, cluster_b, pos_a, pos_b = pair
            
            # Skip if either cluster has 2 connections
            if connection_counts[cluster_a] >= 2 or connection_counts[cluster_b] >= 2:
                continue
                
            # Find path
            path = a_star(self.grid, pos_a, pos_b)
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                weight = distance_m * CENTROID_WIRE_WEIGHT
                
                self.centroid_connections.append({
                    'source': cluster_a,
                    'dest': cluster_b,
                    'path': path,
                    'steps': steps,
                    'distance': distance_m,
                    'weight': weight
                })
                
                connection_counts[cluster_a] += 1
                connection_counts[cluster_b] += 1
                total_connections += 1
                #print(f"Connected {cluster_a} <-> {cluster_b} (Distance: {distance:.1f} grid units)")
                
            # Continue even if some clusters reach max connections
            if total_connections >= 2 * len(clusters):
                break

        print(f"Total centroid connections: {len(self.centroid_connections)}")
        
    def calculate_hpc_position(self, centroid_points):
        """Find optimal HPC position using medoid approach"""
        valid_points = [p for p in centroid_points if self.grid[p[0], p[1]] == 0]
        if not valid_points:
            raise ValueError("No valid HPC positions available")
        
        # Change from 'euclidean' to 'cityblock' for Manhattan distance
        distance_matrix = distance.cdist(valid_points, centroid_points, 'cityblock')
        return valid_points[np.argmin(distance_matrix.sum(axis=1))]

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation

    def process_hpc_paths(self):
        """Process paths from centroids to HPC"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(
                    row['Centroid Coordinate'], self.grid, self.grid_offset
                )
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        try:
            self.hpc_position = self.calculate_hpc_position(list(centroid_coords.values()))
            print(f"HPC established at grid position: {self.hpc_position}")
        except ValueError as e:
            print(f"HPC placement failed: {e}")
            return

        # Find paths from centroids to HPC
        for c_idx, c_pos in centroid_coords.items():
            path = a_star(self.grid, c_pos, self.hpc_position)
            self.centroid_hpc_paths[c_idx] = path if path else []

    def calculate_weights(self):
        """Calculate weights while preserving original coordinates"""
        """Updated weight calculation with battery connections"""
        # Initialize weight trackers
        self.component_weights = defaultdict(float)
        self.hpc_weights = defaultdict(float)
        self.component_distances = defaultdict(float)  # New: Track component distances
        self.hpc_distances = defaultdict(float)       # New: Track HPC distances
        self.hpc_details = []  # Initialize HPC details list
        
        # Original component weight calculation
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values directly from dataframe
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                # Get cluster info from ORIGINAL data
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates for calculations only
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Path calculations
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")
                
                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results with ORIGINAL values
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                
                # Update component distances
                self.component_distances[cluster_id] += distance_m
                self.component_weights[cluster_id] += index_weight

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")

        # Calculate HPC
        # start change for HPC code
        # Initialize HPC details list and processed clusters tracker

        processed_clusters = set()
        
        # Calculate HPC connection weights using Manhattan path distances
        for c_idx, path in self.centroid_hpc_paths.items():
            if not path:
                continue
                
            try:
                cluster_id = self.data.loc[c_idx, 'Cluster ID']
                
                # Skip if we've already processed this cluster
                if cluster_id in processed_clusters:
                    continue
                    
                processed_clusters.add(cluster_id)
                
                # Get grid positions
                centroid_grid = path[0]
                hpc_grid = path[-1]

                # Calculate Manhattan distance for verification
                dx = abs(centroid_grid[0] - hpc_grid[0])
                dy = abs(centroid_grid[1] - hpc_grid[1])
                manhattan_steps = dx + dy

                # Determine path type and actual steps
                line_cells = bresenham_line(centroid_grid, hpc_grid)
                clear_path = all(self.grid[x, y] == 0 for (x, y) in line_cells)

                # # Calculate actual path distance using Manhattan distance
                # distance_mm = 0
                # for i in range(1, len(path)):
                #     x1, y1 = path[i-1]
                #     x2, y2 = path[i]
                    
                #     # Manhattan distance between consecutive points
                #     segment_distance = (abs(x2 - x1) + abs(y2 - y1)) * STEP_SIZE
                #     distance_mm += segment_distance

                if clear_path:
                    steps = manhattan_steps
                    path_type = 'Direct'
                else:
                    path_type = 'A*'
                    steps = len(path) - 1
                    # Ensure steps aren't less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps
                    
                # Calculate distance and weight
                distance_m = (steps * STEP_SIZE) / 1000
                hpc_weight = distance_m * HPC_WIRE_WEIGHT
                    

                
                # Store and accumulate weights (ONCE PER CLUSTER)
                self.hpc_weights[cluster_id] = hpc_weight
                self.hpc_distances[cluster_id] = distance_m

                # Add to HPC details with grid information
                self.hpc_details.append({
                    'Cluster ID': cluster_id,
                    'Centroid Coordinate': self.data.loc[c_idx, 'Centroid Coordinate'],
                    'Start grid': f"{centroid_grid[0]};{centroid_grid[1]}",
                    'HPC Position': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Target grid': f"{hpc_grid[0]};{hpc_grid[1]}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path) - 1,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'HPC Weight (g)': round(hpc_weight, 2)
                })
                
            except Exception as e:
                warnings.warn(f"HPC weight error for cluster {cluster_id}: {e}")

    # end of change for hpc


        # Combine weights
        all_clusters = set(self.component_weights.keys()).union(self.hpc_weights.keys())
        self.cluster_weights = {
            cluster: round(self.component_weights.get(cluster, 0) + self.hpc_weights.get(cluster, 0), 2)
            for cluster in all_clusters
        }

        # Add centroid connection weights
        self.centroid_connection_total = 0
        for conn in self.centroid_connections:
            self.centroid_connection_total += conn['weight']
            # Add to individual cluster weights
            # self.cluster_weights[conn['source']] += conn['weight']/2
            # self.cluster_weights[conn['dest']] += conn['weight']/2

        print(f"Total centroid connection weight: {self.centroid_connection_total:.2f}g")

        
        # === Dynamic Battery connection weights ===
        for cluster_id, path in self.centroid_battery_paths.items():
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                
                # Get weight with validation
                battery_wire_weight = self.cluster_battery_weights.get(cluster_id, BATTERY_WIRE_WEIGHT)  # Ensure string
                
                if battery_wire_weight is None:
                    print(f"⚠️ No battery weight for Cluster {cluster_id}, using fallback 4.26")
                    battery_wire_weight = 4.26
                else:
                    print(f"✅ Using cluster {cluster_id} weight: {battery_wire_weight}g/m")
                    
                self.battery_weights[cluster_id] += distance_m * battery_wire_weight

        # # HPC-battery weight
        # if self.hpc_battery_path:
        #     steps = len(self.hpc_battery_path) - 1
        #     distance_m = (steps * STEP_SIZE) / 1000
        #     self.battery_weights['HPC'] = distance_m * HPC_BATTERY_WIRE_WEIGHT

        # Update total harness weight
        self.total_harness_weight = (
            sum(self.cluster_weights.values()) + 
            self.centroid_connection_total + 
            sum(self.battery_weights.values())  # Now includes all clusters
        )




    def save_to_excel(self, filename):
        """Save results with dynamic column handling and HPC integration"""
        # Create DataFrames
        components_df = pd.DataFrame(self.component_details)
        hpc_connections_df = pd.DataFrame(self.hpc_details) if hasattr(self, 'hpc_details') else pd.DataFrame()
        
        # Define column orders
        components_column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)',
            'Start grid', 'Target grid', 'Path Type','Manhattan Steps','Actual Steps','Used Steps', 
            'Distance (m)', 'Index Weight (g)'
        ]
        
        # Define column orders - ADD CLUSTER ID TO HPC COLUMNS
        hpc_column_order = [
            'Cluster ID', 'Centroid Coordinate', 'HPC Position','Start grid','Target grid', 'Path Type', 'Manhattan Steps',
            'Actual Steps','Used Steps', 'Distance (m)', 'HPC Weight (g)'
        ]
        
        cluster_weight_columns = [
            'Cluster ID','Total Component Distance (m)', 'Centroid-HPC Distance (m)',
            'Component Weight (g)','Centroid-HPC Connection Weight (g)', 'Total Weight (g)'
        ]

        # Add error column if present
        if 'Error' in components_df.columns:
            components_column_order.append('Error')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with pd.ExcelWriter(filename) as writer:
            # 1. Components Sheet (Original format)
            components_df[components_column_order].to_excel(
                writer, sheet_name='Components', index=False)

            # # 2. HPC Connections Sheet (New)
            # if not hpc_connections_df.empty:
            #     hpc_connections_df[hpc_column_order].to_excel(
            #         writer, sheet_name='HPC Connections', index=False)
            
            # start change code
            # 2. HPC Connections Sheet (Updated with Cluster ID)
            if hasattr(self, 'hpc_details') and self.hpc_details:
                hpc_connections_df = pd.DataFrame(self.hpc_details)
                hpc_connections_df[hpc_column_order].to_excel(
                    writer, sheet_name='HPC Connections', index=False)
            else:
                hpc_connections_df = pd.DataFrame()

            # end change code

            # 3. Cluster Weights Sheet (Enhanced)
            cluster_data = []
            for cluster in self.cluster_weights:
                cluster_data.append({
                    'Cluster ID': cluster,
                    'Total Component Distance (m)': round(self.component_distances.get(cluster, 0), 4),
                    'Centroid-HPC Distance (m)': round(self.hpc_distances.get(cluster, 0), 4),
                    'Component Weight (g)': round(self.component_weights.get(cluster, 0), 2),
                    'Centroid-HPC Connection Weight (g)': round(self.hpc_weights.get(cluster, 0), 2),
                    'Total Weight (g)': round(self.cluster_weights.get(cluster, 0), 2)
                })
            
            pd.DataFrame(cluster_data)[cluster_weight_columns].to_excel(
                writer, sheet_name='Cluster Weights', index=False)
            
            
            # 4. Centroid Connections Sheet (Enhanced)
            if self.centroid_connections:
                connection_data = []
                
                # Get original coordinates and grid positions for all clusters
                cluster_info = {}
                unique_clusters = self.data.groupby('Cluster ID').first()
                for cluster_id, row in unique_clusters.iterrows():
                    try:
                        original_coord = row['Centroid Coordinate']
                        grid_pos = float_to_grid(original_coord, self.grid, self.grid_offset)
                        cluster_info[cluster_id] = {
                            'original': original_coord,
                            'grid_x': grid_pos[0],
                            'grid_y': grid_pos[1]
                        }
                    except ValueError:
                        continue

                # Process each connection
                for conn in self.centroid_connections:
                    source = conn['source']
                    dest = conn['dest']
                    steps =  conn['steps']
                    
                    # Get grid coordinates
                    src_grid = (cluster_info[source]['grid_x'], cluster_info[source]['grid_y'])
                    dest_grid = (cluster_info[dest]['grid_x'], cluster_info[dest]['grid_y'])
                    
                    # # Calculate grid distance (Euclidean)
                    # grid_distance = sqrt((src_grid[0]-dest_grid[0])**2 + (src_grid[1]-dest_grid[1])**2)

                    # Calculate grid distance (Manhattan)
                    grid_distance = abs(src_grid[0]-dest_grid[0]) + abs(src_grid[1]-dest_grid[1])

                    
                    connection_data.append({
                        'Source Cluster': source,
                        'Destination Cluster': dest,
                        'Source Original Centroid': cluster_info[source]['original'],
                        'Destination Original Centroid': cluster_info[dest]['original'],
                        'Source Grid': f"{src_grid[0]};{src_grid[1]}",
                        'Destination Grid': f"{dest_grid[0]};{dest_grid[1]}",
                        'Connected Type': f"{source} <-> {dest}",
                        'Grid Distance': round(grid_distance, 2),
                        'steps': steps,
                        'Distance (m)': conn['distance'],
                        'Centroid Connections Weight (g)': conn['weight']
                    })

                centroid_conn_df = pd.DataFrame(connection_data)
                centroid_conn_df = centroid_conn_df[[
                    'Source Cluster', 'Destination Cluster',
                    'Source Original Centroid', 'Destination Original Centroid',
                    'Source Grid', 'Destination Grid', 'Connected Type',
                    'Grid Distance', 'steps', 'Distance (m)',
                    'Centroid Connections Weight (g)'
                ]]
                centroid_conn_df.to_excel(writer, sheet_name='Centroid Connections', index=False)
            
            
        

            # 5. Battery Connections Sheet
            # dynamic nominal power 
            battery_data = []
            processed_clusters = set()

            for cluster_id, path in self.centroid_battery_paths.items():
                if path and self.battery_grid_position and cluster_id not in processed_clusters:
                    try:
                        # Get cluster-specific battery wire weight
                        battery_wire_weight = self.cluster_battery_weights.get(cluster_id)
                        
                        # Fallback with warning if weight missing
                        if battery_wire_weight is None:
                            warnings.warn(f"No battery weight for Cluster {cluster_id}, using 4.26g/m as fallback")
                            battery_wire_weight = BATTERY_WIRE_WEIGHT
                        
                        # Calculate using dynamic weight
                        cluster_data = self.data[self.data['Cluster ID'] == cluster_id].iloc[0]
                        steps = len(path) - 1
                        distance_m = (steps * STEP_SIZE) / 1000
                        weight = distance_m * battery_wire_weight
                        
                        battery_data.append({
                            'Cluster ID': cluster_id,
                            # 'Battery Wire Type':battery_wire_type,
                            'Battery Wire Weight (g/m)': battery_wire_weight,  # Show actual used weight
                            'Centroid Coordinate': cluster_data['Centroid Coordinate'],
                            'BatteryPosition Coordinate': BATTERY_POSITION,
                            'Source Grid': f"{path[0][0]};{path[0][1]}",
                            'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
                            'Steps': steps,
                            'Centroid-BatteryDistance (m)': round(distance_m, 4),
                            'Centroid-BatteryWeight (g)': round(weight, 2)
                        })
                        processed_clusters.add(cluster_id)
                    except Exception as e:
                        print(f"Error processing cluster {cluster_id}: {e}")

            # # HPC-battery connection (keep separate as special case)
            # if self.hpc_battery_path and self.battery_grid_position:
            #     hpc_steps = len(self.hpc_battery_path) - 1
            #     hpc_distance = (hpc_steps * STEP_SIZE) / 1000
            #     battery_data.append({
            #         'Cluster ID': 'HPC',
            #         'Source Coordinate': f"{self.hpc_position[0]};{self.hpc_position[1]}",
            #         'Destination Coordinate': BATTERY_POSITION,
            #         'Source Grid': f"{self.hpc_battery_path[0][0]};{self.hpc_battery_path[0][1]}",
            #         'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
            #         'Steps': hpc_steps,
            #         'Distance (m)': hpc_distance,
            #         'Battery Weight (g)': self.battery_weights['HPC']
            #     })

            pd.DataFrame(battery_data).to_excel(writer, sheet_name='Battery Connections', index=False)

            # 6. Total Weight Sheet (Maintained format)
            total_weight_All_df = pd.DataFrame({
                'Total Component Weight (g)': [sum(self.component_weights.values())],
                'Total Centroid Connections Weight (g)': [self.centroid_connection_total],
                'Total Centroid-HPC Connections Weight (g)': [sum(self.hpc_weights.values())],
                'Total Battery Connections Weight (g)': [sum(self.battery_weights.values())],
                'Total Harness Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_All_df.to_excel(writer, sheet_name='All Weight', index=False)

           # Total weight

            total_weight_df = pd.DataFrame({'Total Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_df.to_excel(writer, sheet_name='Total Weight', index=False)

            # 7. Total Distance Sheet (NEW)
            # Calculate component-to-centroid distances
            total_component_dist = round(sum(self.component_distances.values()), 4)
            
            # Calculate centroid-to-HPC distances
            total_hpc_dist = round(sum(self.hpc_distances.values()), 4)
            
            # Calculate centroid-to-centroid distances
            total_centroid_conn_dist = round(sum(conn['distance'] for conn in self.centroid_connections), 4)
            
            # Calculate battery connection distances
            battery_dist = 0
            # Centroid-battery distances
            for cluster_id, path in self.centroid_battery_paths.items():
                if path:
                    steps = len(path) - 1
                    battery_dist += (steps * STEP_SIZE) / 1000
            # HPC-battery distance
            if self.hpc_battery_path:
                steps = len(self.hpc_battery_path) - 1
                battery_dist += (steps * STEP_SIZE) / 1000
            

            total_distance_All_df = pd.DataFrame({
                'Total Component distance (m)': [total_component_dist],
                'Total Centroid Connections distance (m)': [total_centroid_conn_dist],
                'Total Centroid-HPC Connections distance (m)': [total_hpc_dist],
                'Total Battery Connections distance (m)': [battery_dist],
                'Total Harness distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]
            })
            total_distance_All_df.to_excel(writer, sheet_name='All Distance', index=False)

            # total distance
            total_distance_df= pd.DataFrame({'Total Distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]})
            total_distance_df.to_excel(writer, sheet_name='Total Distance', index=False)


            # 5. Total Weight Sheet (Maintained format)
            # total_component_weight = round(sum(self.cluster_weights.values()), 2)
            # total_weight = total_component_weight + round(self.centroid_connection_total, 2)
            
            # total_df = pd.DataFrame({
            #     'Total Component Weight (g)': [total_component_weight],
            #     'Total Centroid Connections Weight (g)': [round(self.centroid_connection_total, 2)],
            #     'Total Harness Weight (g)': [total_weight]
            # })
            # total_df.to_excel(writer, sheet_name='Total Weight', index=False)

    # Update the visualize method in RoutingProcessor
    def visualize(self, plot_title="Routing Results"):
        return visualize_paths(
            self.grid, self.data, self.paths, self.grid_offset, 
            self.centroid_mapping, self.hpc_position, 
            self.centroid_hpc_paths, self.centroid_connections,
            battery_paths=self.centroid_battery_paths,  # Corrected here
            # hpc_battery_path=self.hpc_battery_path,
            plot_title=plot_title
        )


    def process_paths(self):
        """Find paths and track actual connected centroids"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        for index, row in self.data.iterrows():
            try:
                start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                valid_centroids = []
                
                # Check all centroids for possible connections
                for centroid_idx, centroid in centroid_coords.items():
                    path = a_star(self.grid, start, centroid)
                    if path:
                        valid_centroids.append((len(path), centroid_idx))
                
                if not valid_centroids:
                    self.paths[index] = []
                    self.centroid_mapping[index] = None
                    continue
                
                # Find closest valid centroid
                _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                goal = centroid_coords[closest_idx]
                final_path = a_star(self.grid, start, goal)
                
                self.paths[index] = final_path
                self.centroid_mapping[index] = closest_idx  # Store connected centroid index

            except ValueError as e:
                print(f"Skipping index {index}: {e}")
                self.paths[index] = []
                self.centroid_mapping[index] = None
        
        # Add HPC processing
        self.process_hpc_paths()        
        # Add centroid connections processing
        self.process_centroid_connections()
        
    def process_all(self):
        """Complete processing pipeline"""
        self.process_paths()
        self.process_centroid_connections()
        self.process_hpc_paths()
        self.process_battery_connections()  # New step
        self.calculate_weights()

# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                # output_path = os.path.join(routing_folder, f"routing_{file}")
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # # Process data
                # processor = RoutingProcessor(restricted_file, data_path)
                # processor.process_paths()
                # processor.calculate_weights()
                # processor.save_to_excel(output_path)
                # # fig = processor.visualize(plot_title=base_name)
                # fig = processor.visualize(plot_title=base_name)

                # Process with new pipeline
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_all()  # Includes battery connections
                processor.save_to_excel(output_path)
                
                # Generate visualization first
                fig = processor.visualize(plot_title=f"{base_name} with Battery Connections")

                # Save and close the figure
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {

        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/kmeans_clustersWireTypeCentroidsToBattery",
        "routing_folder": "./data/routing/centroidsToBatteryPowerWire",
        "plot_folder": "./data/save_plot/centroidsToBatteryPowerWire_plots"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")

# ==================================================== end centroids To Battery without HPC-battery =================================


# ======================================11.0 Generate Wire Type based on Dynamic Traffic scenarios ====================================

# # 11.0 Automatically Generate Wire Type:
# - Dynamic Traffic scenarios
# - based on (CurrentCarryingCapacity>=MaxDemanedCurrent) and Battery_Zone_Connection =Yes 
# - get wire_weight[g/m]

import os
import pandas as pd
from pathlib import Path

# Paths
input_owncloud_path = './data/extract_data/extracted_data.xlsx'
input_driving_power_folder = './data/merge_data'
output_folder = './data/GetWireTypeDrivingScenarios'

# Create output directory if not exist
os.makedirs(output_folder, exist_ok=True)

# Load wire characteristics
wire_characteristics = pd.read_excel(input_owncloud_path, sheet_name='wire_characteristics_data')
wire_characteristics = wire_characteristics[
    ['wire_type', 'wire_weight', 'CO2_Emission', 'wire_CurrentCarryingCapacity[A]', 'Battery_Zone_Connection']
].rename(columns={
    'wire_CurrentCarryingCapacity[A]': 'CurrentCarryingCapacity[A]',
    'wire_weight': 'wire_weight[g/m]'
})

# Iterate through each drivingPower file
for filename in os.listdir(input_driving_power_folder):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(input_driving_power_folder, filename)
        
        try:
            # Load data and filter for PowerSupply='Yes'
            driving_power_data = pd.read_excel(file_path, sheet_name='Merge_DrivingScenariosCurrent')
            
            # Filter for power supply components only
            driving_power_data = driving_power_data[
                driving_power_data['PowerSupply'].str.upper() == 'YES'
            ]
            
            if driving_power_data.empty:
                print(f"⚠️ Skipping {filename}: No power supply components found")
                continue

            # Verify required columns
            required_cols = [
                'Cluster ID', 
                'SummerSunnyDayCityIdleCurrent', 
                'WinterSunnyDayHighwayCurrent', 
                'WinterRainyNightCityIdleCurrent', 
                'SummerRainyNightHighwayCurrent'
            ]
            missing_cols = [col for col in required_cols if col not in driving_power_data.columns]
            if missing_cols:
                print(f"⚠️ Skipping {filename}: Missing columns {', '.join(missing_cols)}")
                continue

            # Convert current columns to numeric
            current_cols = required_cols[1:]  # Exclude Cluster ID
            for col in current_cols:
                driving_power_data[col] = pd.to_numeric(driving_power_data[col], errors='coerce').fillna(0)

            # Group by Cluster ID and sum currents
            grouped = driving_power_data.groupby('Cluster ID').agg({
                'SummerSunnyDayCityIdleCurrent': 'sum',
                'WinterSunnyDayHighwayCurrent': 'sum',
                'WinterRainyNightCityIdleCurrent': 'sum',
                'SummerRainyNightHighwayCurrent': 'sum'
            }).reset_index().rename(columns={
                'SummerSunnyDayCityIdleCurrent': 'DemanedCurrentD1',
                'WinterSunnyDayHighwayCurrent': 'DemanedCurrentD2',
                'WinterRainyNightCityIdleCurrent': 'DemanedCurrentD3',
                'SummerRainyNightHighwayCurrent': 'DemanedCurrentD4'
            })

            # Calculate maximum demanded current
            grouped['MaxDemanedCurrent'] = grouped[['DemanedCurrentD1', 'DemanedCurrentD2', 
                                                    'DemanedCurrentD3', 'DemanedCurrentD4']].max(axis=1)

            # Enhanced wire selection with fallback
            def find_appropriate_wire(current):
                # Get battery wires
                battery_wires = wire_characteristics[
                    wire_characteristics['Battery_Zone_Connection'] == 'Yes'
                ].copy()
                
                # Find sufficient wires
                sufficient_wires = battery_wires[
                    battery_wires['CurrentCarryingCapacity[A]'] >= current
                ]
                
                if not sufficient_wires.empty:
                    # Get smallest sufficient wire
                    wire = sufficient_wires.nsmallest(1, 'CurrentCarryingCapacity[A]').iloc[0]
                    return pd.Series([
                        wire['wire_type'],
                        wire['CurrentCarryingCapacity[A]'],
                        wire['wire_weight[g/m]'],
                        'Sufficient'
                    ])
                
                # If no sufficient wire, find closest lower capacity wire
                if not battery_wires.empty:
                    # Calculate difference below required current
                    battery_wires['capacity_diff'] = current - battery_wires['CurrentCarryingCapacity[A]']
                    # Filter for wires below current and get closest
                    below_wires = battery_wires[battery_wires['CurrentCarryingCapacity[A]'] < current]
                    if not below_wires.empty:
                        wire = below_wires.nsmallest(1, 'capacity_diff').iloc[0]
                        return pd.Series([
                            wire['wire_type'],
                            wire['CurrentCarryingCapacity[A]'],
                            wire['wire_weight[g/m]'],
                            'Insufficient'
                        ])
                
                # Default if no wires found
                return pd.Series(['No Suitable Wire', None, None, 'Error'])

            # Apply wire selection
            result_cols = grouped['MaxDemanedCurrent'].apply(find_appropriate_wire)
            result_cols.columns = ['battery_wire_type', 'CurrentCarryingCapacity[A]', 
                                  'battery_wire_weight[g/m]', 'Wire_Status']
            grouped = pd.concat([grouped, result_cols], axis=1)

            # Save results
            new_filename = f"{Path(filename).stem.replace('routing_', '')}.xlsx"
            output_path = os.path.join(output_folder, new_filename)
            
            with pd.ExcelWriter(output_path) as writer:
                grouped.to_excel(writer, sheet_name='GetWireType', index=False)

        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")

print("✅ All files processed and updated with safe wire sizing under Battery Zone Connection = Yes.")

#================================================= end ====================================

#========================================== 12. 0 Merge K-Means and driving scenerios wire type ====================================

# # 12.0 Merge data based driving scenerio demanded current for kmeans_clustersBattery
# - Merge k-means and Wire type define demanded current data
# - this dataset is used for calculating cluster weight for driving scenerio demanded current
# - Wire type define automatically based on driving scenerio demanded current


import os
import pandas as pd

# Define paths
cluster_input_folder = './data/routing/kmeans_clustersAvoidRZ'
wire_input_folder = './data/GetWireTypeDrivingScenarios'
output_folder = './data/kmeans_clustersWireTypeDrivingScenariosToBattery'



# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all cluster result files
for filename in os.listdir(cluster_input_folder):
    if filename.endswith('.xls') or filename.endswith('.xlsx'):
        cluster_file_path = os.path.join(cluster_input_folder, filename)
        wire_file_path = os.path.join(wire_input_folder, filename)

        # Check if corresponding wire_typeDynamic file exists
        if not os.path.exists(wire_file_path):
            print(f"⚠️  Skipping {filename}: No matching file in GetWireType.")
            continue

        try:
            # Read sheets
            cluster_df = pd.read_excel(cluster_file_path, sheet_name='Cluster_Data')
            wire_df = pd.read_excel(wire_file_path, sheet_name='GetWireType')
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

        # Ensure required column exists
        if 'Cluster ID' not in cluster_df.columns or 'Cluster ID' not in wire_df.columns:
            print(f"⚠️  Skipping {filename}: Missing 'Cluster ID' in one of the sheets.")
            continue

        # Select needed columns from wire data
        wire_df_filtered = wire_df[['Cluster ID', 'battery_wire_type', 'battery_wire_weight[g/m]']]

        # Merge on Cluster ID
        merged_df = pd.merge(cluster_df, wire_df_filtered, on='Cluster ID', how='left')

        # Save merged result
        output_path = os.path.join(output_folder, filename)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='Cluster_Data', index=False)

        print(f"✅ Merged and saved: {filename}")

print("🏁 All matched files processed and saved to 'kmeans_clustersWireTypeDrivingScenariosToBattery'.")

# ====================================== end  =================================================================


#================================ 13.0 start centroidHpc To Battery Driving Traffic scenerios ========================================
# ### 13.0 Battery Connection
# - Centroids to Battery
# - HPC to Battery
# - Battery Position: 98.0, 18.0
# - Battery Connection wire weight: Dynamic wire type
# - use kmeans_clustersBattery data




# Add to top with other imports
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from math import sqrt
import os
import warnings
from math import sqrt
from itertools import combinations
from scipy.spatial import distance



STEP_SIZE = STEP_SIZE  # Distance per grid step in mm
# Update the constants near other constants
# FLRY 2x0,13: 2,04 g/m or FLRY 2x0,35 : 4,26 g/m
CENTROID_WIRE_WEIGHT = CENTROID_WIRE_WEIGHT  # centroid-to-centroid connections; CAN-FD lines. They can operate up to 8 Mbit/s, typically 2 Mbit/s. standard ISO 11898. FLRY 2x0,13 or FLRY 2x0,35
# Add new constants near other constants
HPC_WIRE_WEIGHT = HPC_WIRE_WEIGHT  # 6g/m for centroid-to-HPC connections; Simpler cars can operate with 100 Mbit/s Ethernet. For this purpose, FLKS9Y 2x0,13 cables are used

# ---------------------------- Add Battery Constants ----------------------------
BATTERY_POSITION = BATTERY_POSITION # Real-world coordinates
BATTERY_WIRE_WEIGHT = BATTERY_WIRE_WEIGHT  # 4,26 g/m for battery connections; FLRY 0,35 : 4,26 g/m
HPC_BATTERY_WIRE_WEIGHT = HPC_BATTERY_WIRE_WEIGHT  # 6g/m for HPC-battery connections (example value):	FLKS9Y 2x0.13 (Ethernet)

# ---------------------------- Original Grid Handling ----------------------------

def create_dynamic_grid(data, restricted_coords):
    """Create grid including battery position"""
    # Convert battery position to tuple first
    battery_tuple = tuple(map(float, BATTERY_POSITION.split(';')))
    
    all_coords = restricted_coords + \
        [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate'] if ';' in str(c)] + \
        [tuple(map(float, str(c).split(';'))) for c in data['Centroid Coordinate'] if ';' in str(c)] + \
        [battery_tuple]  # Add battery position as proper tuple
    
    # Rest of the function remains unchanged
    x_values = [x for x, y in all_coords]
    y_values = [y for x, y in all_coords]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    grid_rows = int(round(max_x - min_x)) + 1
    grid_cols = int(round(max_y - min_y)) + 1
    
    grid = np.zeros((grid_rows, grid_cols))
    for x, y in restricted_coords:
        x_rounded = int(round(x - min_x))
        y_rounded = int(round(y - min_y))
        if 0 <= x_rounded < grid_rows and 0 <= y_rounded < grid_cols:
            grid[x_rounded, y_rounded] = 1
    return grid, (min_x, min_y)

def float_to_grid(coord, grid, offset):
    """Convert real coordinates to 1mm grid cells"""
    try:
        x, y = tuple(map(float, str(coord).split(';')))
        min_x, min_y = offset
        x_int = int(round(x - min_x))
        y_int = int(round(y - min_y))

        if not (0 <= x_int < grid.shape[0] and 0 <= y_int < grid.shape[1]):
            raise ValueError(f"Coordinate {coord} maps to out-of-grid ({x_int}, {y_int})")

        if grid[x_int, y_int] == 1:
            valid_coords = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
            if not valid_coords:
                raise ValueError("No valid coordinates available")
            nearest = min(valid_coords, key=lambda c: (c[0]-x_int)**2 + (c[1]-y_int)**2)
            return nearest
        return (x_int, y_int)
    except Exception as e:
        raise ValueError(f"Error processing {coord}: {e}")


def visualize_paths(grid, data, paths, offset, centroid_mapping, hpc_position=None, 
                    centroid_hpc_paths=None, centroid_connections=None, 
                    battery_paths=None, hpc_battery_path=None,  
                    plot_title="Routing Visualization"):
    """Enhanced visualization with battery connections"""
    fig = plt.figure(figsize=(15, 15))
    min_x, min_y = offset
    
    # 1. Plot restricted zones
    plt.imshow(grid.T, cmap='Greys_r', alpha=0.7, origin='lower',
              extent=[min_x, min_x + grid.shape[0], 
                     min_y, min_y + grid.shape[1]])
    
    # 2. Create color palette for clusters
    unique_clusters = sorted(data['Cluster ID'].unique())
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters))) if len(unique_clusters) > 0 else []
    
    # 3. Plot component-to-centroid paths
    plotted_clusters = set()
    for index, path in paths.items():
        if path and index in centroid_mapping:
            try:
                cluster_id = data.iloc[centroid_mapping[index]]['Cluster ID']
                color = colors[unique_clusters.index(cluster_id)]
                
                if cluster_id not in plotted_clusters:
                    label = f'Cluster {cluster_id}'
                    plotted_clusters.add(cluster_id)
                else:
                    label = None
                
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color=color, 
                        linewidth=3, alpha=0.9, label=label)
            except (IndexError, KeyError):
                continue

    # 4. Plot components
    try:
        components = [tuple(map(float, str(c).split(';'))) for c in data['Index Coordinate']]
        plt.scatter(
            [c[0] for c in components], [c[1] for c in components],
            color='royalblue', s=120, marker='o', edgecolor='black',
            linewidth=1.5, label='Components', zorder=4
        )
    except ValueError:
        pass

    # 5. Plot centroids with cluster IDs
    try:
        centroids_df = data.groupby('Cluster ID').first()
        centroids = []
        cluster_ids = []
        for cluster_id, row in centroids_df.iterrows():
            coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
            centroids.append(coord)
            cluster_ids.append(cluster_id)
        
        plt.scatter(
            [c[0] for c in centroids], [c[1] for c in centroids],
            color='gold', s=200, marker='X', edgecolor='black',
            linewidth=2, label='Centroids', zorder=5
        )
        
        # Add cluster ID text labels
        for (x, y), cluster_id in zip(centroids, cluster_ids):
            plt.text(x, y, str(cluster_id), 
                    fontsize=10, weight='bold', color='black',
                    ha='center', va='center', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    except Exception as e:
        print(f"Error plotting centroids: {e}")

    # 6. Plot HPC connections and node
    if hpc_position and centroid_hpc_paths:
        for path in centroid_hpc_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, 'purple', linestyle=':', 
                        linewidth=3, alpha=0.8, label='Centroid-HPC', zorder=2)
        
        hpc_x = hpc_position[0] + min_x
        hpc_y = hpc_position[1] + min_y
        plt.scatter(hpc_x, hpc_y, color='red', s=300, marker='*',
                   edgecolor='black', label='HPC', zorder=6)

    # 7. Plot centroid-to-centroid connections
    if centroid_connections:
        for conn in centroid_connections:
            path = conn['path']
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                plt.plot(x_coords, y_coords, color='lime', linestyle='--', 
                        linewidth=2.5, alpha=0.9, label='Centroid Links', zorder=3)

    # 8. Plot battery connections
    battery_plotted = False
    if battery_paths:
        for path in battery_paths.values():
            if path:
                x_coords = [p[0] + min_x for p in path]
                y_coords = [p[1] + min_y for p in path]
                label = 'Centroid-Battery' if not battery_plotted else None
                plt.plot(x_coords, y_coords, 'darkorange', linestyle='-.', 
                        linewidth=2.5, alpha=0.9, label=label, zorder=3)
                if not battery_plotted:
                    battery_plotted = True

    # 9. Plot Battery node
    battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
    plt.scatter(battery_x, battery_y, color='lime', s=350, marker='P',
               edgecolor='black', linewidth=2, label='Battery', zorder=7)

    # 10. Create unified legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    
    legend_order = [
        'Components', 'Centroids', 'Centroid Links', 'HPC',
        'Centroid-HPC', 'Battery', 'Centroid-Battery'
    ] + [f'Cluster {cid}' for cid in unique_clusters]
    
    ordered_handles = [unique[l] for l in legend_order if l in unique]
    ordered_labels = [l for l in legend_order if l in unique]
    
    plt.legend(ordered_handles, ordered_labels, 
               loc='upper left', 
               bbox_to_anchor=(1.01, 1),  # Move legend outside
               borderaxespad=0.,
               frameon=True,
               title="Legend", title_fontsize=10,
               fontsize=8, handlelength=2.0,
               borderpad=1.2, labelspacing=1.2)

    plt.xlabel("X Coordinate (mm)", fontsize=12)
    plt.ylabel("Y Coordinate (mm)", fontsize=12)
    plt.title(plot_title, fontsize=14, pad=20)
    plt.grid(visible=False)
    plt.tight_layout()
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Reserve 15% space on right

    
    return fig

# ---------------------------- Helper Functions ----------------------------
def load_restricted_zones(file_path, sheet_name):
    """Load restricted coordinates from Excel file"""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return [tuple(map(int, coord.split(';'))) for coord in data['Restricted_Coordinate'] if ';' in str(coord)]

def bresenham_line(start, end):
    """Generate cells along the line from start to end using Bresenham's algorithm"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    line_cells = []
    
    while True:
        line_cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return line_cells

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
    
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None

# ---------------------------- Processor Class ----------------------------
class RoutingProcessor:
    def __init__(self, restricted_file, data_file):
        self.restricted_coords = load_restricted_zones(restricted_file, "restricted_data")
        self.data = pd.read_excel(data_file, sheet_name="Cluster_Data")
 
        # # # New: Store battery wire weights per cluster
        # # self.cluster_battery_weights = self.data.groupby('Cluster ID')['battery_wire_weight[g/m]'].first().to_dict()

        # # Debug: Print column names
        # print("\n=== Columns in Data ===")
        # print(self.data.columns.tolist())
        
        # # Debug: Print sample battery weights
        # print("\n=== Battery Weight Samples ===")
        # if 'battery_wire_weight[g/m]' in self.data.columns:
        #     print(self.data[['Cluster ID', 'battery_wire_weight[g/m]']].head(3))
        # else:
        #     print("Column 'battery_wire_weight[g/m]' not found!")
        
        
        # Load weights (now with string IDs)
        self.cluster_battery_weights = self.data.groupby('Cluster ID')['battery_wire_weight[g/m]'].first().to_dict()
        
        # # Debug: Print loaded weights
        # print("\n=== Loaded Battery Weights ===")
        # print(self.cluster_battery_weights)

        self.grid, self.grid_offset = create_dynamic_grid(self.data, self.restricted_coords)
        self.paths = {}
        self.component_details = []
        self.cluster_weights = {}
        self.centroid_mapping = {}
        self.hpc_position = None
        self.centroid_hpc_paths = {}
        self.centroid_connections = [] 
        self.battery_position = None
        self.centroid_battery_paths = {}
        self.hpc_battery_path = None
        self.battery_weights = defaultdict(float)
        self.battery_paths = {}
        # self.hpc_battery_path = None
        # self.battery_weights = defaultdict(float)


    def process_battery_connections(self):
        """Connect centroids and HPC to battery"""
        try:
            battery_x, battery_y = map(float, BATTERY_POSITION.split(';'))
            battery_coord = f"{battery_x};{battery_y}"
            # Store both original and grid positions
            self.battery_position = (battery_x, battery_y)  # Original coordinates
            battery_grid = float_to_grid(battery_coord, self.grid, self.grid_offset)
            self.battery_grid_position = battery_grid  # Grid coordinates
        except ValueError as e:
            print(f"Battery placement failed: {e}")
            self.battery_position = None
            self.battery_grid_position = None
            return

        # Connect centroids to battery using grid position
        cluster_centroids = self.data.groupby('Cluster ID').first()['Centroid Coordinate']
        
        self.centroid_battery_paths = {}
        for cluster_id, centroid_coord in cluster_centroids.items():
            try:
                centroid_pos = float_to_grid(centroid_coord, self.grid, self.grid_offset)
                path = a_star(self.grid, centroid_pos, self.battery_grid_position)
                self.centroid_battery_paths[cluster_id] = path
            except Exception as e:
                print(f"Skipping cluster {cluster_id} battery connection: {e}")

        # # Connect HPC to battery using grid position
        # if self.hpc_position and self.battery_grid_position:
        #     try:
        #         self.hpc_battery_path = a_star(self.grid, self.hpc_position, self.battery_grid_position)
        #     except Exception as e:
        #         print(f"HPC-battery connection failed: {e}")
    

    def process_centroid_connections(self):
        """Improved centroid connection logic with better logging"""
        #print("\n=== Processing Centroid Connections ===")
        
        # Get valid cluster centroids
        unique_clusters = self.data.groupby('Cluster ID').first()
        cluster_centroids = {}
        for cluster_id, row in unique_clusters.iterrows():
            try:
                original_coord = tuple(map(float, str(row['Centroid Coordinate']).split(';')))
                grid_pos = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
                #print(f"Cluster {cluster_id}: Original {original_coord} -> Grid {grid_pos}")
                cluster_centroids[cluster_id] = grid_pos
            except ValueError as e:
                print(f"Skipping cluster {cluster_id}: {str(e)}")
                continue

        # Generate all possible pairs
        clusters = list(cluster_centroids.keys())
        pairs = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_a = clusters[i]
                cluster_b = clusters[j]
                pos_a = cluster_centroids[cluster_a]
                pos_b = cluster_centroids[cluster_b]
                distance = sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                pairs.append((distance, cluster_a, cluster_b, pos_a, pos_b))

        # Sort by distance
        pairs.sort(key=lambda x: x[0])
        
        connection_counts = defaultdict(int)
        self.centroid_connections = []
        total_connections = 0
        
        for pair in pairs:
            distance, cluster_a, cluster_b, pos_a, pos_b = pair
            
            # Skip if either cluster has 2 connections
            if connection_counts[cluster_a] >= 2 or connection_counts[cluster_b] >= 2:
                continue
                
            # Find path
            path = a_star(self.grid, pos_a, pos_b)
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                weight = distance_m * CENTROID_WIRE_WEIGHT
                
                self.centroid_connections.append({
                    'source': cluster_a,
                    'dest': cluster_b,
                    'path': path,
                    'steps': steps,
                    'distance': distance_m,
                    'weight': weight
                })
                
                connection_counts[cluster_a] += 1
                connection_counts[cluster_b] += 1
                total_connections += 1
                #print(f"Connected {cluster_a} <-> {cluster_b} (Distance: {distance:.1f} grid units)")
                
            # Continue even if some clusters reach max connections
            if total_connections >= 2 * len(clusters):
                break

        print(f"Total centroid connections: {len(self.centroid_connections)}")
        
    def calculate_hpc_position(self, centroid_points):
        """Find optimal HPC position using medoid approach"""
        valid_points = [p for p in centroid_points if self.grid[p[0], p[1]] == 0]
        if not valid_points:
            raise ValueError("No valid HPC positions available")
        
        # Change from 'euclidean' to 'cityblock' for Manhattan distance
        distance_matrix = distance.cdist(valid_points, centroid_points, 'cityblock')
        return valid_points[np.argmin(distance_matrix.sum(axis=1))]

    # Keep all other methods (process_paths, calculate_weights, etc.) unchanged from previous implementation

    def process_hpc_paths(self):
        """Process paths from centroids to HPC"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(
                    row['Centroid Coordinate'], self.grid, self.grid_offset
                )
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        try:
            self.hpc_position = self.calculate_hpc_position(list(centroid_coords.values()))
            print(f"HPC established at grid position: {self.hpc_position}")
        except ValueError as e:
            print(f"HPC placement failed: {e}")
            return

        # Find paths from centroids to HPC
        for c_idx, c_pos in centroid_coords.items():
            path = a_star(self.grid, c_pos, self.hpc_position)
            self.centroid_hpc_paths[c_idx] = path if path else []

    def calculate_weights(self):
        """Calculate weights while preserving original coordinates"""
        """Updated weight calculation with battery connections"""
        # Initialize weight trackers
        self.component_weights = defaultdict(float)
        self.hpc_weights = defaultdict(float)
        self.component_distances = defaultdict(float)  # New: Track component distances
        self.hpc_distances = defaultdict(float)       # New: Track HPC distances
        self.hpc_details = []  # Initialize HPC details list
        
        # Original component weight calculation
        for idx, row in self.data.iterrows():
            try:
                # Preserve original values directly from dataframe
                original_index = row['Index']
                original_index_coord = row['Index Coordinate']
                original_centroid_coord = row['Centroid Coordinate']

                # Get connected centroid info
                connected_centroid_idx = self.centroid_mapping.get(idx)
                if connected_centroid_idx is None:
                    raise ValueError("No valid path to any centroid")
                
                # Get cluster info from ORIGINAL data
                cluster_id = self.data.loc[connected_centroid_idx, 'Cluster ID']
                actual_centroid_coord = self.data.loc[connected_centroid_idx, 'Centroid Coordinate']

                # Convert coordinates for calculations only
                start = float_to_grid(original_index_coord, self.grid, self.grid_offset)
                centroid = float_to_grid(actual_centroid_coord, self.grid, self.grid_offset)

                # Path calculations
                path = self.paths.get(idx, [])
                if not path:
                    raise ValueError("No valid path exists")
                
                # Calculate Manhattan distance for verification
                dx = abs(centroid[0] - start[0])
                dy = abs(centroid[1] - start[1])
                manhattan_steps = dx + dy

                # Determine path type and steps
                line_cells = bresenham_line(start, centroid)
                clear_path = all(self.grid[x][y] == 0 for (x, y) in line_cells)
                
                if clear_path:
                    path_type = 'Direct'
                    steps = manhattan_steps  # Use Manhattan distance for direct paths
                else:
                    path_type = 'A*'
                    steps = len(path) - 1  # Actual path length for A* paths
                    
                    # Validate A* path steps - shouldn't be less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps  # Ensure minimum steps

                # Calculate weights
                distance_mm = steps * STEP_SIZE
                distance_m = distance_mm / 1000
                wire_weight = row['index Wire Weight(g/m)']
                index_weight = distance_m * wire_weight

                # Store results with ORIGINAL values
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Original Centroid': original_centroid_coord,
                    'Connected Centroid': actual_centroid_coord,
                    'Cluster ID': cluster_id,
                    'index Wire Weight(g/m)': wire_weight,
                    'Start grid': f"{start}",
                    'Target grid': f"{centroid}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path)-1 if path else 0,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'Index Weight (g)': round(index_weight, 2)
                })

                # Update cluster weights
                self.cluster_weights[cluster_id] = self.cluster_weights.get(cluster_id, 0) + index_weight
                
                # Update component distances
                self.component_distances[cluster_id] += distance_m
                self.component_weights[cluster_id] += index_weight

            except Exception as e:
                self.component_details.append({
                    'Index': original_index,
                    'Index Coordinate': original_index_coord,
                    'Error': str(e)
                })
                warnings.warn(f"Error processing index {original_index}: {e}")

        # Calculate HPC
        # start change for HPC code
        # Initialize HPC details list and processed clusters tracker

        processed_clusters = set()
        
        # Calculate HPC connection weights using Manhattan path distances
        for c_idx, path in self.centroid_hpc_paths.items():
            if not path:
                continue
                
            try:
                cluster_id = self.data.loc[c_idx, 'Cluster ID']
                
                # Skip if we've already processed this cluster
                if cluster_id in processed_clusters:
                    continue
                    
                processed_clusters.add(cluster_id)
                
                # Get grid positions
                centroid_grid = path[0]
                hpc_grid = path[-1]

                # Calculate Manhattan distance for verification
                dx = abs(centroid_grid[0] - hpc_grid[0])
                dy = abs(centroid_grid[1] - hpc_grid[1])
                manhattan_steps = dx + dy

                # Determine path type and actual steps
                line_cells = bresenham_line(centroid_grid, hpc_grid)
                clear_path = all(self.grid[x, y] == 0 for (x, y) in line_cells)

                # # Calculate actual path distance using Manhattan distance
                # distance_mm = 0
                # for i in range(1, len(path)):
                #     x1, y1 = path[i-1]
                #     x2, y2 = path[i]
                    
                #     # Manhattan distance between consecutive points
                #     segment_distance = (abs(x2 - x1) + abs(y2 - y1)) * STEP_SIZE
                #     distance_mm += segment_distance

                if clear_path:
                    steps = manhattan_steps
                    path_type = 'Direct'
                else:
                    path_type = 'A*'
                    steps = len(path) - 1
                    # Ensure steps aren't less than Manhattan distance
                    if steps < manhattan_steps:
                        steps = manhattan_steps
                    
                # Calculate distance and weight
                distance_m = (steps * STEP_SIZE) / 1000
                hpc_weight = distance_m * HPC_WIRE_WEIGHT
                    

                
                # Store and accumulate weights (ONCE PER CLUSTER)
                self.hpc_weights[cluster_id] = hpc_weight
                self.hpc_distances[cluster_id] = distance_m

                # Add to HPC details with grid information
                self.hpc_details.append({
                    'Cluster ID': cluster_id,
                    'Centroid Coordinate': self.data.loc[c_idx, 'Centroid Coordinate'],
                    'Start grid': f"{centroid_grid[0]};{centroid_grid[1]}",
                    'HPC Position': f"{self.hpc_position[0]};{self.hpc_position[1]}",
                    'Target grid': f"{hpc_grid[0]};{hpc_grid[1]}",
                    'Path Type': path_type,
                    'Manhattan Steps': manhattan_steps,
                    'Actual Steps': len(path) - 1,
                    'Used Steps': steps,
                    'Distance (m)': round(distance_m, 4),
                    'HPC Weight (g)': round(hpc_weight, 2)
                })
                
            except Exception as e:
                warnings.warn(f"HPC weight error for cluster {cluster_id}: {e}")

    # end of change for hpc



        # Combine weights
        all_clusters = set(self.component_weights.keys()).union(self.hpc_weights.keys())
        self.cluster_weights = {
            cluster: round(self.component_weights.get(cluster, 0) + self.hpc_weights.get(cluster, 0), 2)
            for cluster in all_clusters
        }

        # Add centroid connection weights
        self.centroid_connection_total = 0
        for conn in self.centroid_connections:
            self.centroid_connection_total += conn['weight']
            # Add to individual cluster weights
            # self.cluster_weights[conn['source']] += conn['weight']/2
            # self.cluster_weights[conn['dest']] += conn['weight']/2

        print(f"Total centroid connection weight: {self.centroid_connection_total:.2f}g")

        
        # === Dynamic Battery connection weights ===
        for cluster_id, path in self.centroid_battery_paths.items():
            if path:
                steps = len(path) - 1
                distance_m = (steps * STEP_SIZE) / 1000
                
                # Get weight with validation
                battery_wire_weight = self.cluster_battery_weights.get(cluster_id, BATTERY_WIRE_WEIGHT)  # Ensure string
                
                if battery_wire_weight is None:
                    print(f"⚠️ No battery weight for Cluster {cluster_id}, using fallback 4.26")
                    battery_wire_weight = 4.26
                else:
                    print(f"✅ Using cluster {cluster_id} weight: {battery_wire_weight}g/m")
                    
                self.battery_weights[cluster_id] += distance_m * battery_wire_weight

        # # HPC-battery weight
        # if self.hpc_battery_path:
        #     steps = len(self.hpc_battery_path) - 1
        #     distance_m = (steps * STEP_SIZE) / 1000
        #     self.battery_weights['HPC'] = distance_m * HPC_BATTERY_WIRE_WEIGHT

        # Update total harness weight
        self.total_harness_weight = (
            sum(self.cluster_weights.values()) + 
            self.centroid_connection_total + 
            sum(self.battery_weights.values())  # Now includes all clusters
        )




    def save_to_excel(self, filename):
        """Save results with dynamic column handling and HPC integration"""
        # Create DataFrames
        components_df = pd.DataFrame(self.component_details)
        hpc_connections_df = pd.DataFrame(self.hpc_details) if hasattr(self, 'hpc_details') else pd.DataFrame()
        
        # Define column orders
        components_column_order = [
            'Index', 'Index Coordinate', 'Original Centroid',
            'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)',
            'Start grid', 'Target grid', 'Path Type','Manhattan Steps','Actual Steps','Used Steps', 
            'Distance (m)', 'Index Weight (g)'
        ]
        
        # Define column orders - ADD CLUSTER ID TO HPC COLUMNS
        hpc_column_order = [
            'Cluster ID', 'Centroid Coordinate', 'HPC Position','Start grid','Target grid', 'Path Type', 'Manhattan Steps',
            'Actual Steps','Used Steps', 'Distance (m)', 'HPC Weight (g)'
        ]
        
        cluster_weight_columns = [
            'Cluster ID','Total Component Distance (m)', 'Centroid-HPC Distance (m)',
            'Component Weight (g)','Centroid-HPC Connection Weight (g)', 'Total Weight (g)'
        ]

        # Add error column if present
        if 'Error' in components_df.columns:
            components_column_order.append('Error')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with pd.ExcelWriter(filename) as writer:
            # 1. Components Sheet (Original format)
            components_df[components_column_order].to_excel(
                writer, sheet_name='Components', index=False)

            # # 2. HPC Connections Sheet (New)
            # if not hpc_connections_df.empty:
            #     hpc_connections_df[hpc_column_order].to_excel(
            #         writer, sheet_name='HPC Connections', index=False)

            # start change code
            # 2. HPC Connections Sheet (Updated with Cluster ID)
            if hasattr(self, 'hpc_details') and self.hpc_details:
                hpc_connections_df = pd.DataFrame(self.hpc_details)
                hpc_connections_df[hpc_column_order].to_excel(
                    writer, sheet_name='HPC Connections', index=False)
            else:
                hpc_connections_df = pd.DataFrame()

            # end change code

            # 3. Cluster Weights Sheet (Enhanced)
            cluster_data = []
            for cluster in self.cluster_weights:
                cluster_data.append({
                    'Cluster ID': cluster,
                    'Total Component Distance (m)': round(self.component_distances.get(cluster, 0), 4),
                    'Centroid-HPC Distance (m)': round(self.hpc_distances.get(cluster, 0), 4),
                    'Component Weight (g)': round(self.component_weights.get(cluster, 0), 2),
                    'Centroid-HPC Connection Weight (g)': round(self.hpc_weights.get(cluster, 0), 2),
                    'Total Weight (g)': round(self.cluster_weights.get(cluster, 0), 2)
                })
            
            pd.DataFrame(cluster_data)[cluster_weight_columns].to_excel(
                writer, sheet_name='Cluster Weights', index=False)
            
            
            # 4. Centroid Connections Sheet (Enhanced)
            if self.centroid_connections:
                connection_data = []
                
                # Get original coordinates and grid positions for all clusters
                cluster_info = {}
                unique_clusters = self.data.groupby('Cluster ID').first()
                for cluster_id, row in unique_clusters.iterrows():
                    try:
                        original_coord = row['Centroid Coordinate']
                        grid_pos = float_to_grid(original_coord, self.grid, self.grid_offset)
                        cluster_info[cluster_id] = {
                            'original': original_coord,
                            'grid_x': grid_pos[0],
                            'grid_y': grid_pos[1]
                        }
                    except ValueError:
                        continue

                # Process each connection
                for conn in self.centroid_connections:
                    source = conn['source']
                    dest = conn['dest']
                    steps =  conn['steps']
                    
                    # Get grid coordinates
                    src_grid = (cluster_info[source]['grid_x'], cluster_info[source]['grid_y'])
                    dest_grid = (cluster_info[dest]['grid_x'], cluster_info[dest]['grid_y'])
                    
                    # # Calculate grid distance (Euclidean)
                    # grid_distance = sqrt((src_grid[0]-dest_grid[0])**2 + (src_grid[1]-dest_grid[1])**2)

                    # Calculate grid distance (Manhattan)
                    grid_distance = abs(src_grid[0]-dest_grid[0]) + abs(src_grid[1]-dest_grid[1])

                    
                    connection_data.append({
                        'Source Cluster': source,
                        'Destination Cluster': dest,
                        'Source Original Centroid': cluster_info[source]['original'],
                        'Destination Original Centroid': cluster_info[dest]['original'],
                        'Source Grid': f"{src_grid[0]};{src_grid[1]}",
                        'Destination Grid': f"{dest_grid[0]};{dest_grid[1]}",
                        'Connected Type': f"{source} <-> {dest}",
                        'Grid Distance': round(grid_distance, 2),
                        'steps': steps,
                        'Distance (m)': conn['distance'],
                        'Centroid Connections Weight (g)': conn['weight']
                    })

                centroid_conn_df = pd.DataFrame(connection_data)
                centroid_conn_df = centroid_conn_df[[
                    'Source Cluster', 'Destination Cluster',
                    'Source Original Centroid', 'Destination Original Centroid',
                    'Source Grid', 'Destination Grid', 'Connected Type',
                    'Grid Distance', 'steps', 'Distance (m)',
                    'Centroid Connections Weight (g)'
                ]]
                centroid_conn_df.to_excel(writer, sheet_name='Centroid Connections', index=False)
            
            
        

            # 5. Battery Connections Sheet
            # for dynamic traffic scenerios
            
            battery_data = []
            processed_clusters = set()

            for cluster_id, path in self.centroid_battery_paths.items():
                if path and self.battery_grid_position and cluster_id not in processed_clusters:
                    try:
                        # Get cluster-specific battery wire weight
                        battery_wire_weight = self.cluster_battery_weights.get(cluster_id)
                        
                        # Fallback with warning if weight missing
                        if battery_wire_weight is None:
                            warnings.warn(f"No battery weight for Cluster {cluster_id}, using 4.26g/m as fallback")
                            battery_wire_weight = 4.26
                        
                        # Calculate using dynamic weight
                        cluster_data = self.data[self.data['Cluster ID'] == cluster_id].iloc[0]
                        steps = len(path) - 1
                        distance_m = (steps * STEP_SIZE) / 1000
                        weight = distance_m * battery_wire_weight
                        
                        battery_data.append({
                            'Cluster ID': cluster_id,
                            'Battery Wire Weight (g/m)': battery_wire_weight,  # Show actual used weight
                            'Centroid Coordinate': cluster_data['Centroid Coordinate'],
                            'BatteryPosition Coordinate': BATTERY_POSITION,
                            'Source Grid': f"{path[0][0]};{path[0][1]}",
                            'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
                            'Steps': steps,
                            'Centroid-BatteryDistance (m)': round(distance_m, 4),
                            'Centroid-BatteryWeight (g)': round(weight, 2)
                        })
                        processed_clusters.add(cluster_id)
                    except Exception as e:
                        print(f"Error processing cluster {cluster_id}: {e}")

            # # HPC-battery connection (keep separate as special case)
            # if self.hpc_battery_path and self.battery_grid_position:
            #     hpc_steps = len(self.hpc_battery_path) - 1
            #     hpc_distance = (hpc_steps * STEP_SIZE) / 1000
            #     battery_data.append({
            #         'Cluster ID': 'HPC',
            #         'Source Coordinate': f"{self.hpc_position[0]};{self.hpc_position[1]}",
            #         'Destination Coordinate': BATTERY_POSITION,
            #         'Source Grid': f"{self.hpc_battery_path[0][0]};{self.hpc_battery_path[0][1]}",
            #         'Destination Grid': f"{self.battery_grid_position[0]};{self.battery_grid_position[1]}",
            #         'Steps': hpc_steps,
            #         'Distance (m)': hpc_distance,
            #         'Battery Weight (g)': self.battery_weights['HPC']
            #     })

            pd.DataFrame(battery_data).to_excel(writer, sheet_name='Battery Connections', index=False)

            # 6. Total Weight Sheet (Maintained format)
            total_weight_All_df = pd.DataFrame({
                'Total Component Weight (g)': [sum(self.component_weights.values())],
                'Total Centroid Connections Weight (g)': [self.centroid_connection_total],
                'Total Centroid-HPC Connections Weight (g)': [sum(self.hpc_weights.values())],
                'Total Battery Connections Weight (g)': [sum(self.battery_weights.values())],
                'Total Harness Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_All_df.to_excel(writer, sheet_name='All Weight', index=False)

           # Total weight

            total_weight_df = pd.DataFrame({'Total Weight (g)': [
                    sum(self.component_weights.values()) + 
                    self.centroid_connection_total + 
                    sum(self.battery_weights.values())+
                    sum(self.hpc_weights.values())
                ]
            })
            total_weight_df.to_excel(writer, sheet_name='Total Weight', index=False)

            # 7. Total Distance Sheet (NEW)
            # Calculate component-to-centroid distances
            total_component_dist = round(sum(self.component_distances.values()), 4)
            
            # Calculate centroid-to-HPC distances
            total_hpc_dist = round(sum(self.hpc_distances.values()), 4)
            
            # Calculate centroid-to-centroid distances
            total_centroid_conn_dist = round(sum(conn['distance'] for conn in self.centroid_connections), 4)
            
            # Calculate battery connection distances
            battery_dist = 0
            # Centroid-battery distances
            for cluster_id, path in self.centroid_battery_paths.items():
                if path:
                    steps = len(path) - 1
                    battery_dist += (steps * STEP_SIZE) / 1000
            # HPC-battery distance
            if self.hpc_battery_path:
                steps = len(self.hpc_battery_path) - 1
                battery_dist += (steps * STEP_SIZE) / 1000
            

            total_distance_All_df = pd.DataFrame({
                'Total Component distance (m)': [total_component_dist],
                'Total Centroid Connections distance (m)': [total_centroid_conn_dist],
                'Total Centroid-HPC Connections distance (m)': [total_hpc_dist],
                'Total Battery Connections distance (m)': [battery_dist],
                'Total Harness distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]
            })
            total_distance_All_df.to_excel(writer, sheet_name='All Distance', index=False)

            # total distance
            total_distance_df= pd.DataFrame({'Total Distance (m)': [round(total_component_dist + total_hpc_dist + total_centroid_conn_dist + battery_dist, 4)]})
            total_distance_df.to_excel(writer, sheet_name='Total Distance', index=False)


            # 5. Total Weight Sheet (Maintained format)
            # total_component_weight = round(sum(self.cluster_weights.values()), 2)
            # total_weight = total_component_weight + round(self.centroid_connection_total, 2)
            
            # total_df = pd.DataFrame({
            #     'Total Component Weight (g)': [total_component_weight],
            #     'Total Centroid Connections Weight (g)': [round(self.centroid_connection_total, 2)],
            #     'Total Harness Weight (g)': [total_weight]
            # })
            # total_df.to_excel(writer, sheet_name='Total Weight', index=False)

    # Update the visualize method in RoutingProcessor
    def visualize(self, plot_title="Routing Results"):
        return visualize_paths(
            self.grid, self.data, self.paths, self.grid_offset, 
            self.centroid_mapping, self.hpc_position, 
            self.centroid_hpc_paths, self.centroid_connections,
            battery_paths=self.centroid_battery_paths,  # Corrected here
            # hpc_battery_path=self.hpc_battery_path,
            plot_title=plot_title
        )


    def process_paths(self):
        """Find paths and track actual connected centroids"""
        centroid_coords = {}
        for idx, row in self.data.iterrows():
            try:
                centroid_coords[idx] = float_to_grid(row['Centroid Coordinate'], self.grid, self.grid_offset)
            except ValueError as e:
                print(f"Skipping centroid {idx}: {e}")
        
        for index, row in self.data.iterrows():
            try:
                start = float_to_grid(row['Index Coordinate'], self.grid, self.grid_offset)
                valid_centroids = []
                
                # Check all centroids for possible connections
                for centroid_idx, centroid in centroid_coords.items():
                    path = a_star(self.grid, start, centroid)
                    if path:
                        valid_centroids.append((len(path), centroid_idx))
                
                if not valid_centroids:
                    self.paths[index] = []
                    self.centroid_mapping[index] = None
                    continue
                
                # Find closest valid centroid
                _, closest_idx = min(valid_centroids, key=lambda x: x[0])
                goal = centroid_coords[closest_idx]
                final_path = a_star(self.grid, start, goal)
                
                self.paths[index] = final_path
                self.centroid_mapping[index] = closest_idx  # Store connected centroid index

            except ValueError as e:
                print(f"Skipping index {index}: {e}")
                self.paths[index] = []
                self.centroid_mapping[index] = None
        
        # Add HPC processing
        self.process_hpc_paths()        
        # Add centroid connections processing
        self.process_centroid_connections()
        
    def process_all(self):
        """Complete processing pipeline"""
        self.process_paths()
        self.process_centroid_connections()
        self.process_hpc_paths()
        self.process_battery_connections()  # New step
        self.calculate_weights()

# ---------------------------- Batch Processing ----------------------------
def process_multiple_files(restricted_file, cluster_folder, routing_folder, plot_folder):
    """Process multiple files with proper plot handling"""
    os.makedirs(routing_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    for file in os.listdir(cluster_folder):
        if file.endswith(".xlsx"):
            try:
                # Prepare paths
                base_name = os.path.splitext(file)[0]
                data_path = os.path.join(cluster_folder, file)
                # output_path = os.path.join(routing_folder, f"routing_{file}")
                output_path = os.path.join(routing_folder, f"{file}")
                plot_path = os.path.join(plot_folder, f"{base_name}.png")

                # # Process data
                # processor = RoutingProcessor(restricted_file, data_path)
                # processor.process_paths()
                # processor.calculate_weights()
                # processor.save_to_excel(output_path)
                # # fig = processor.visualize(plot_title=base_name)
                # fig = processor.visualize(plot_title=base_name)

                # Process with new pipeline
                processor = RoutingProcessor(restricted_file, data_path)
                processor.process_all()  # Includes battery connections
                processor.save_to_excel(output_path)
                
                # Generate visualization first
                fig = processor.visualize(plot_title=f"{base_name} with Battery Connections")

                # Save and close the figure
                if fig is not None:
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"Successfully saved: {plot_path}")
                else:
                    print(f"Skipping plot for {file} - no figure generated")

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    # Configure paths
    config = {

        "restricted_file": "./data/extract_data/extracted_data.xlsx",
        "cluster_folder": "./data/kmeans_clustersWireTypeDrivingScenariosToBattery",
        "routing_folder": "./data/routing/centroidHpcToBatteryDynamicWire",
        "plot_folder": "./data/save_plot/centroidHpcToBatteryDynamicWire_plots"
    }
    
    process_multiple_files(**config)
    print("Batch processing completed")

# =====================================end centroidHpcToBatteryDynamicWire =================================================


# ============================= 15.0 extract all weight and distance ===========================================================
# # 15.0 # Extract kmeans-clustering Analysis results
# - extract from all folder


import os
import pandas as pd
import re

# Input and output paths
parent_dir = './data/routing'
output_dir = './data/kmeans_resultAnalysis'
os.makedirs(output_dir, exist_ok=True)

# File names
output_weights_path = os.path.join(output_dir, 'combined_cluster_weights.xlsx')
output_distances_path = os.path.join(output_dir, 'combined_cluster_distances.xlsx')

# Helper function to extract and accumulate data by sheet type
def collect_application_data(sheet_name):
    collected = {}

    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        summary_data = {}

        for filename in os.listdir(folder_path):
            if filename.endswith('.xls') or filename.endswith('.xlsx'):
                match = re.match(r"cluster_results_(Low|Mid|High)_Application_(\d+)", filename)
                if not match:
                    continue

                app_type = match.group(1) + "_Application"
                zone_cluster = match.group(2)
                file_path = os.path.join(folder_path, filename)

                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                except Exception as e:
                    print(f"⚠️ Skipping {filename} in {folder} (sheet '{sheet_name}'): {e}")
                    continue

                value = df.select_dtypes(include='number').values.flatten()
                if len(value) == 0:
                    continue
                total_value = value[0]

                if zone_cluster not in summary_data:
                    summary_data[zone_cluster] = {}
                summary_data[zone_cluster][app_type] = total_value

        if summary_data:
            collected[folder] = summary_data

    return collected

# Collect both weights and distances
weights_collected = collect_application_data('Total Weight')
distances_collected = collect_application_data('Total Distance')

# Write weights
with pd.ExcelWriter(output_weights_path, engine='openpyxl') as writer:
    for folder, data in weights_collected.items():
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df.rename(columns={'index': 'Zone_Cluster'}, inplace=True)
        for col in ['Low_Application', 'Mid_Application', 'High_Application']:
            if col not in df.columns:
                df[col] = None
        df = df[['Zone_Cluster', 'Low_Application', 'Mid_Application', 'High_Application']]
        df['Zone_Cluster'] = df['Zone_Cluster'].astype(int)
        df = df.sort_values(by='Zone_Cluster')
        df.to_excel(writer, sheet_name=folder[:31], index=False)

# Write distances
with pd.ExcelWriter(output_distances_path, engine='openpyxl') as writer:
    for folder, data in distances_collected.items():
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df.rename(columns={'index': 'Zone_Cluster'}, inplace=True)
        for col in ['Low_Application', 'Mid_Application', 'High_Application']:
            if col not in df.columns:
                df[col] = None
        df = df[['Zone_Cluster', 'Low_Application', 'Mid_Application', 'High_Application']]
        df['Zone_Cluster'] = df['Zone_Cluster'].astype(int)
        df = df.sort_values(by='Zone_Cluster')
        df.to_excel(writer, sheet_name=folder[:31], index=False)

print(f"✅ Saved both files:\n - {output_weights_path}\n - {output_distances_path}")


# =============================end extract data==========================================================================


# ============================ 16.0 start extract data from centroidsToBatteryPowerWire =====================================

# - Extract data from centroids-To-Battery-Power-Wire model
# - 


import pandas as pd
import os
from glob import glob
import re

def aggregate_centroid_data(input_dir, output_file):
    weight_columns = [
        "Total Component Weight (g)",
        "Total Centroid Connections Weight (g)",
        "Total Centroid-HPC Connections Weight (g)",
        "Total Battery Connections Weight (g)",
        "Total Harness Weight (g)"
    ]

    distance_columns = [
        "Total Component distance (m)",
        "Total Centroid Connections distance (m)",
        "Total Centroid-HPC Connections distance (m)",
        "Total Battery Connections distance (m)",
        "Total Harness distance (m)"
    ]

    results = {
        "Weight_Low": [],
        "Weight_Mid": [],
        "Weight_High": [],
        "Distance_Low": [],
        "Distance_Mid": [],
        "Distance_High": [],
    }

    file_paths = glob(os.path.join(input_dir, "cluster_results_*.xlsx"))

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            # Extract application level
            if "Low_Application" in file_name:
                app = "Low"
            elif "Mid_Application" in file_name:
                app = "Mid"
            elif "High_Application" in file_name:
                app = "High"
            else:
                continue  # skip non-matching

            # Extract cluster ID (e.g., ..._1.xlsx → 1)
            match = re.search(r'_(\d+)\.xlsx$', file_name)
            cluster_id = int(match.group(1)) if match else None

            # Read sheets
            weight_df = pd.read_excel(file_path, sheet_name="All Weight", nrows=1, usecols="A:E")
            distance_df = pd.read_excel(file_path, sheet_name="All Distance", nrows=1, usecols="A:E")

            # Clean column names and convert comma-decimal to float
            weight_df.columns = [col.strip() for col in weight_df.columns]
            distance_df.columns = [col.strip() for col in distance_df.columns]
            weight_df = weight_df.applymap(lambda x: str(x).replace(",", ".")).astype(float)
            distance_df = distance_df.applymap(lambda x: str(x).replace(",", ".")).astype(float)

            # Add Cluster ID
            weight_df.insert(0, "Cluster ID", cluster_id)
            distance_df.insert(0, "Cluster ID", cluster_id)

            results[f"Weight_{app}"].append(weight_df.iloc[0])
            results[f"Distance_{app}"].append(distance_df.iloc[0])

        except Exception as e:
            print(f"⚠ Error processing file {file_name}: {e}")

    # Write final Excel output
    with pd.ExcelWriter(output_file) as writer:
        for key, rows in results.items():
            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=key, index=False)

# Run the function
input_directory = './data/routing/centroidsToBatteryPowerWire'
output_excel = './data/routing/aggregated_centroid_results.xlsx'
aggregate_centroid_data(input_directory, output_excel)
print("✅ Aggregation complete with Cluster IDs.")

# ============================ end extract data from centroidsToBatteryPowerWire =====================================

# ======================= 17. Merge weight and distance data for training AI to get wire type based on driving for specific cluster================

# 17.0 Merge data for Dynamic wire type get for Power Analysis using Matlab, AI
# - from topology/data/merge_data 
# - topology/data/routing/centroidsToBatteryPowerWire
# - topology/data/GetWireTypeCentroids

import pandas as pd
import os
import re

# Define base directories
base_dir = './data'
merge_data_dir = os.path.join(base_dir, 'merge_data')
routing_dir = os.path.join(base_dir, 'routing/centroidHpcToBatteryDynamicWire')
getwire_dir = os.path.join(base_dir, 'GetWireTypeDrivingScenarios')
output_dir = os.path.join(base_dir, 'merge_dynamicWire')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Define column order (same as original)
desired_columns = [
    'Index', 'PinoutNr', 'PinoutSignal', 'PowerSupply', 'Signal',
    'Wire_Gauge_[mm²]@85°C_@12V', 'wire_length_[g/m]', 'Nominal_Power[W]',
    'Nominal_Current_@12V [A]', 'SummerSunnyDayCityIdleTraffic',
    'SummerSunnyDayCityIdleCurrent', 'WinterSunnyDayHighwayTraffic',
    'WinterSunnyDayHighwayCurrent', 'WinterRainyNightCityIdleTraffic',
    'WinterRainyNightCityIdleCurrent', 'SummerRainyNightHighwayTraffic',
    'SummerRainyNightHighwayCurrent', 'Index Coordinate', 
    'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)', 
    'Used Steps', 'Distance (m)', 'Index Weight (g)', 
    'BatteryPosition Coordinate','DemanedCurrentD1','DemanedCurrentD2',
    'DemanedCurrentD3','DemanedCurrentD4','MaxDemanedCurrent',
    'CurrentCarryingCapacity[A]','battery_wire_type', 
    'battery_wire_weight[g/m]', 'Centroid-BatteryDistance (m)',
    'Centroid-BatteryWeight (g)'
]

# Get all merge data files
merge_files = [f for f in os.listdir(merge_data_dir) 
              if f.startswith('cluster_results_') and f.endswith('.xlsx')]

for file in merge_files:
    try:
        # Construct file paths
        merge_data_path = os.path.join(merge_data_dir, file)
        routing_path = os.path.join(routing_dir, file)
        getwire_path = os.path.join(getwire_dir, file)
        output_path = os.path.join(output_dir, file)

        # Check if required files exist
        missing_files = []
        if not os.path.exists(routing_path):
            missing_files.append(routing_path)
        if not os.path.exists(getwire_path):
            missing_files.append(getwire_path)
            
        if missing_files:
            print(f"Skipping {file}: Missing files - {', '.join(missing_files)}")
            continue

        # Read data from each source
        df_merge = pd.read_excel(merge_data_path, sheet_name='Merge_DrivingScenariosCurrent')
        df_battery = pd.read_excel(routing_path, sheet_name='Battery Connections')
        df_wiretype = pd.read_excel(getwire_path, sheet_name='GetWireType')

        # Merge dataframes
        merged_df = pd.merge(df_merge, df_battery, on='Cluster ID', how='left')
        merged_df = pd.merge(merged_df, df_wiretype, on='Cluster ID', how='left')

        # Handle missing columns
        for col in desired_columns:
            if col not in merged_df.columns:
                merged_df[col] = None  # Add missing columns as empty

        # Reorder columns
        merged_df = merged_df[desired_columns]

        # Save results
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='dynamicWire', index=False)
            
        print(f"Successfully processed: {file}")

    except Exception as e:
        print(f"Error processing {file}: {str(e)}")

# print("\nProcessing completed. Check output directory for results.")

#=============================18. start Extract data for driving_scenarios power analysis ==========================
import pandas as pd
import os
import glob

# Define directories
input_dir = './data/merge_dynamicWire'
output_dir = './data/merge_driving_scenarios'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Define columns to extract with new names
column_mapping = {
    'Cluster ID': 'Cluster ID',
    'Connected Centroid': 'Connected Centroid',
    'BatteryPosition Coordinate': 'BatteryPosition Coordinate',
    'MaxDemanedCurrent': 'MaxDemanedCurrent',
    'CurrentCarryingCapacity[A]': 'CurrentCarryingCapacity[A]',
    'battery_wire_type': 'battery_wire_type',
    'battery_wire_weight[g/m]': 'battery_wire_weight[g/m]',
    'Centroid-BatteryDistance (m)': 'Centroid-BatteryDistance(m)',  # Rename to 'distance'
    'Centroid-BatteryWeight (g)': 'Centroid-BatteryWeight(g)'
}

# Original column names needed for extraction
source_columns = list(column_mapping.keys())

# Process all Excel files
for excel_file in glob.glob(os.path.join(input_dir, '*.xlsx')):
    try:
        # Read Excel file
        df = pd.read_excel(excel_file, sheet_name='dynamicWire')
        
        # Check for required columns
        missing_cols = [col for col in source_columns if col not in df.columns]
        if missing_cols:
            print(f"Skipping {os.path.basename(excel_file)}: Missing columns - {', '.join(missing_cols)}")
            continue
        
        # Extract and deduplicate by Cluster ID
        extracted_df = df[source_columns].drop_duplicates(
            subset='Cluster ID', 
            keep='first'
        )
        
        # Rename columns according to our mapping
        extracted_df = extracted_df.rename(columns=column_mapping)
        
        # Create output filename
        base_name = os.path.basename(excel_file)
        output_path = os.path.join(output_dir, base_name)  # Keep .xlsx extension
        
        # Save as Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            extracted_df.to_excel(writer, sheet_name='driving_scenarios', index=False)
        
        print(f"Saved {len(extracted_df)} unique clusters to: {base_name}")
        
    except Exception as e:
        print(f"Error processing {excel_file}: {str(e)}")

# print("\nProcessing complete!")
# print(f"Excel files saved to: {output_dir}")
# print("Each file contains unique entries based on 'Cluster ID'")

#=============================18. end Extract data for driving_scenarios power analysis ==========================

# ======================================== 19. start Wires Characteristics data load ===============================================================

import pandas as pd
import os

# Define the folder to save plots
data_folder = './data/wires_wharacteristics/'
os.makedirs(data_folder, exist_ok=True)  # Create the folder if it doesn't exist
 
# Load the Excel file
#file_path = 'C:/Users/mhossain.HHN/Documents/PhD/wiring_harness_car/HarnessAnalysis/harness_tool_final/data/owncloud_data/2025-04-30_EDAG_KROSCHU_EE_FEATURELIST_KI4BOARDNET_v3.xlsx'
file_path = './data/owncloud_data/2025-04-30_EDAG_KROSCHU_EE_FEATURELIST_KI4BOARDNET_v3.xlsx'

# --- Extract components_data ---
wires_characteristics = pd.read_excel(file_path, sheet_name='Wires Characteristics', skiprows=9)

# --- Extract wire_characteristics_data ---
wire_characteristics_data = pd.DataFrame({
    'wire_type': wires_characteristics.iloc[:, 0],
    'description': wires_characteristics.iloc[:, 1],
    'standard': wires_characteristics.iloc[:, 2],
    'battery_Zone_Connection': wires_characteristics.iloc[:, 4],
    'Conductor_Type': wires_characteristics.iloc[:, 5],
    'conductor_cross_section[mm²]': wires_characteristics.iloc[:, 6],
    'Conductor_Max_Electrical_Resistance': wires_characteristics.iloc[:, 7],
    'conductor_metal_weight[g/m]': wires_characteristics.iloc[:, 8],
    'insulation_type': wires_characteristics.iloc[:, 9],
    'insulation_max_outer_diameter[mm]': wires_characteristics.iloc[:, 10],
    'insulation_thickness[mm]': wires_characteristics.iloc[:, 11],
    'assembly_min_bending_radius[mm]': wires_characteristics.iloc[:, 12],
    'Current_Carrying_Capacity': wires_characteristics.iloc[:, 13],
    'Assembly_Aprox_Total_Weight': wires_characteristics.iloc[:, 14],
    'assembly_aprox_CO2_emission[g/m]': wires_characteristics.iloc[:, 15],
})

# # --- Save to CSV instead of Excel ---
# wire_characteristics_data.to_csv("./data/wire_database.csv", index=False)


output_file = os.path.join(data_folder, "wire_characteristics_data.xlsx")
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    wire_characteristics_data.to_excel(writer, sheet_name='wire_characteristics_data', index=False)


print("Data extraction completed and saved to 'wire_database.xlsx'.")

# ===================================================19. start Wires Characteristics data load ========================================================

#====================================================20. start merge_driving_scenarios_wire_characteristics_data==============

import pandas as pd
import os
import glob

# Define directories
input_dir = './data/merge_driving_scenarios'
wire_char_path = './data/wires_wharacteristics/wire_characteristics_data.xlsx'
output_dir = './data/merge_driving_scenarios_wire'

# Create output directory with verbose logging
print(f"Creating output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory exists: {os.path.exists(output_dir)}")
print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")

# Verify wire characteristics file exists
if not os.path.exists(wire_char_path):
    print(f"CRITICAL ERROR: Wire characteristics file not found at: {wire_char_path}")
    print("Please verify the file exists at this path.")
    exit()
else:
    print(f"Wire characteristics file found: {wire_char_path}")

# Load wire characteristics data
try:
    wire_df = pd.read_excel(wire_char_path, sheet_name='wire_characteristics_data')
    print("Successfully loaded wire characteristics data")
    
    # Select and rename columns for merging
    wire_df = wire_df[['wire_type', 'Conductor_Type', 'Conductor_Max_Electrical_Resistance']]
    wire_df.rename(columns={'wire_type': 'battery_wire_type'}, inplace=True)
    print(f"Wire characteristics columns: {wire_df.columns.tolist()}")
    print(f"Number of wire types: {len(wire_df)}")
    
except Exception as e:
    print(f"Error loading wire characteristics: {str(e)}")
    exit()

# List input files
input_files = glob.glob(os.path.join(input_dir, '*.xlsx'))
print(f"\nFound {len(input_files)} files in input directory:")
for f in input_files:
    print(f"- {os.path.basename(f)}")

# Process files
for excel_file in input_files:
    try:
        file_name = os.path.basename(excel_file)
        print(f"\nProcessing: {file_name}")
        
        # Read driving scenario file
        scenario_df = pd.read_excel(excel_file, sheet_name='driving_scenarios')
        print(f"Original columns: {scenario_df.columns.tolist()}")
        print(f"Rows in original: {len(scenario_df)}")
        
        # Check for battery_wire_type column
        if 'battery_wire_type' not in scenario_df.columns:
            print(f"ERROR: 'battery_wire_type' column not found in {file_name}")
            continue
            
        # Merge with wire characteristics
        merged_df = pd.merge(
            scenario_df,
            wire_df,
            on='battery_wire_type',
            how='left'
        )
        print(f"Rows after merge: {len(merged_df)}")
        print(f"New columns: {merged_df.columns.tolist()}")
        
        # Create output path
        output_path = os.path.join(output_dir, file_name)
        print(f"Output path: {output_path}")
        
        # Save with explicit file path
        merged_df.to_excel(output_path, sheet_name='driving_scenarios', index=False)
        print(f"File saved: {output_path}")
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"Verification: File exists at {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")
        else:
            print(f"ERROR: File not created at {output_path}")
        
    except Exception as e:
        print(f"Error processing {excel_file}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\nProcessing complete!")
print(f"Merged files should be in: {output_dir}")

#====================================================20. end merge_driving_scenarios_wire_characteristics_data==============

#====================================================21. start merge_NominalPowerData__wire_characteristics_data==============

# This is used for calculation voltage drop for comparison journal 2

import pandas as pd
import os
import glob

# Define directories
input_dir = './data/merge_driving_scenarios'
wire_char_path = './data/wires_wharacteristics/wire_characteristics_data.xlsx'
output_dir = './data/merge_driving_scenarios_wire'

# Create output directory with verbose logging
print(f"Creating output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory exists: {os.path.exists(output_dir)}")
print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")

# Verify wire characteristics file exists
if not os.path.exists(wire_char_path):
    print(f"CRITICAL ERROR: Wire characteristics file not found at: {wire_char_path}")
    print("Please verify the file exists at this path.")
    exit()
else:
    print(f"Wire characteristics file found: {wire_char_path}")

# Load wire characteristics data
try:
    wire_df = pd.read_excel(wire_char_path, sheet_name='wire_characteristics_data')
    print("Successfully loaded wire characteristics data")
    
    # Select and rename columns for merging
    wire_df = wire_df[['wire_type', 'Conductor_Type', 'Conductor_Max_Electrical_Resistance']]
    wire_df.rename(columns={'wire_type': 'battery_wire_type'}, inplace=True)
    print(f"Wire characteristics columns: {wire_df.columns.tolist()}")
    print(f"Number of wire types: {len(wire_df)}")
    
except Exception as e:
    print(f"Error loading wire characteristics: {str(e)}")
    exit()

# List input files
input_files = glob.glob(os.path.join(input_dir, '*.xlsx'))
print(f"\nFound {len(input_files)} files in input directory:")
for f in input_files:
    print(f"- {os.path.basename(f)}")

# Process files
for excel_file in input_files:
    try:
        file_name = os.path.basename(excel_file)
        print(f"\nProcessing: {file_name}")
        
        # Read driving scenario file
        scenario_df = pd.read_excel(excel_file, sheet_name='driving_scenarios')
        print(f"Original columns: {scenario_df.columns.tolist()}")
        print(f"Rows in original: {len(scenario_df)}")
        
        # Check for battery_wire_type column
        if 'battery_wire_type' not in scenario_df.columns:
            print(f"ERROR: 'battery_wire_type' column not found in {file_name}")
            continue
            
        # Merge with wire characteristics
        merged_df = pd.merge(
            scenario_df,
            wire_df,
            on='battery_wire_type',
            how='left'
        )
        print(f"Rows after merge: {len(merged_df)}")
        print(f"New columns: {merged_df.columns.tolist()}")
        
        # Create output path
        output_path = os.path.join(output_dir, file_name)
        print(f"Output path: {output_path}")
        
        # Save with explicit file path
        merged_df.to_excel(output_path, sheet_name='driving_scenarios', index=False)
        print(f"File saved: {output_path}")
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"Verification: File exists at {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")
        else:
            print(f"ERROR: File not created at {output_path}")
        
    except Exception as e:
        print(f"Error processing {excel_file}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\nProcessing complete!")
print(f"Merged files should be in: {output_dir}")

#====================================================20. end merge_driving_scenarios_wire_characteristics_data==============

# ======================================10.6 Merge weight and distance data for training AI to get wire type or Nominal Power =====================

# # 10.6 Merge data for Nominal Power wire select
# we are going to merge the data for analysis Power harness design in Matlab, AI
# - from topology/data/merge_data
# - topology/data/routing/centroidsToBatteryPowerWire
# - topology/data/GetWireTypeCentroids

import pandas as pd
import os

# Define cluster types and directories
cluster_types = ['High', 'Mid', 'Low']
base_dir = './data'
merge_data_dir = os.path.join(base_dir, 'merge_data')
routing_dir = os.path.join(base_dir, 'routing/centroidsToBatteryPowerWire')
getwire_dir = os.path.join(base_dir, 'GetWireTypeCentroids')
output_dir = os.path.join(base_dir, 'merge_nominalWire')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the desired column order
desired_columns = [
    'Index', 'PinoutNr', 'PinoutSignal', 'PowerSupply', 'Signal',
    'Wire_Gauge_[mm²]@85°C_@12V', 'wire_length_[g/m]', 'Nominal_Power[W]',
    'Nominal_Current_@12V [A]', #'SummerSunnyDayCityIdleTraffic',
    #'SummerSunnyDayCityIdleCurrent', 'WinterSunnyDayHighwayTraffic',
    #'WinterSunnyDayHighwayCurrent', 'WinterRainyNightCityIdleTraffic',
    #'WinterRainyNightCityIdleCurrent', 'SummerRainyNightHighwayTraffic',
    #'SummerRainyNightHighwayCurrent', 
    'Index Coordinate', #'Original Centroid',
    'Connected Centroid', 'Cluster ID', 'index Wire Weight(g/m)', #'Start grid',
    #'Target grid', 'Path Type', 'Manhattan Steps', 'Actual Steps', 
    'Used Steps',
    'Distance (m)', 'Index Weight (g)', 'BatteryPosition Coordinate',
    #'Source Grid', 'Destination Grid', 'Steps',
    'battery_wire_type', 'battery_wire_weight[g/m]', #'CurrentCarryingCapacity[A]', 
    'Centroid-BatteryDistance (m)',
    'Centroid-BatteryWeight (g)'
    
]

for cluster in cluster_types:
    # Construct file paths
    merge_data_path = os.path.join(merge_data_dir, f'cluster_results_{cluster}_Application_12.xlsx')
    routing_path = os.path.join(routing_dir, f'cluster_results_{cluster}_Application_12.xlsx')
    getwire_path = os.path.join(getwire_dir, f'cluster_results_{cluster}_Application_12.xlsx')
    output_path = os.path.join(output_dir, f'cluster_results_{cluster}_Application_12.xlsx')

    try:
        # Read data from each sheet
        df_merge = pd.read_excel(merge_data_path, sheet_name='Merge_DrivingScenariosCurrent')
        df_battery = pd.read_excel(routing_path, sheet_name='Battery Connections')
        df_wiretype = pd.read_excel(getwire_path, sheet_name='CentroidsGetWireType')

        # Merge dataframes on 'Cluster ID'
        merged_df = pd.merge(df_merge, df_battery, on='Cluster ID', how='left')
        merged_df = pd.merge(merged_df, df_wiretype, on='Cluster ID', how='left')

        # Reorder columns
        missing_cols = [col for col in desired_columns if col not in merged_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {cluster} cluster: {missing_cols}")
            # Add missing columns with NaN values
            for col in missing_cols:
                merged_df[col] = None
        merged_df = merged_df[desired_columns]

        # Save to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            merged_df.to_excel(writer, sheet_name='nominalWire', index=False)
        print(f"Successfully processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {cluster} cluster: {e}")


# ==================================================================end merge data for AI training=================================================