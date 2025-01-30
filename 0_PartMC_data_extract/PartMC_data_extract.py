"""
This script processes simulation outputs from the PartMC model, focusing on gas-phase and aerosol-phase properties 
within the particle diameter range of 100–700 nm. The analysis is conducted over 1,000 different scenarios, 
with each scenario containing multiple NetCDF (.nc) files representing time-resolved simulation data.

Key objectives of this script include:
1. Extracting gas-phase mixing ratios (e.g., O3, CO, NO, NOx, CH3OH) and aerosol-phase mass concentrations (e.g., BC, OA, NH4).
2. Calculating the mixing state index (Chi) for aerosols based on their chemical composition.
3. Compiling environmental data such as temperature, relative humidity (RH), and time from each simulation.
4. Aggregating all extracted data into a structured tabular format for downstream analysis.

The output of this script is a comprehensive CSV file containing scenario-wise data, which is essential for 
studies involving aerosol chemical composition, mixing state, and their interactions with environmental variables.
The processed data is stored in a file named `PartMC_data_100To700nm.csv` for further exploration and analysis.

This script is designed for reproducibility and extensibility, enabling seamless integration with other 
data analysis pipelines for aerosol research.

Author: Fei Jiang, The University of Manchester
"""


import pmcpy  # A library to handle PartMC simulation output files
import xarray as xr  # For working with NetCDF data files
import numpy as np  # For numerical operations
import pandas as pd  # For creating and manipulating tabular data
import os  # For file and directory operations


# Define a function to count the number of .nc files in a given directory
def count_nc_files(directory):
    """Count the number of NetCDF (.nc) files in a specified directory.

    Args:
        directory (str): The directory path to search for .nc files.

    Returns:
        int: The total number of .nc files found in the directory.
    """
    return sum(1 for file in os.listdir(directory) if file.endswith('.nc'))

# Initialize an empty DataFrame to store the final results
df = pd.DataFrame(columns=[
    'Scenario_ID', 'DayofYear', 'Time(hr)', 'O3 (ppb)', 'CO (ppb)', 
    'NO (ppb)', 'NOx (ppb)', 'CH3OH (ppb)', 'C2H6 (ppb)', 'ETH (ppb)', 
    'TOL(ppb)', 'XYL (ppb)', 'ALD2 (ppb)', 'AONE (ppb)', 'PAR (ppb)', 
    'OLET (ppb)', 'Temperature(K)', 'RH', 'BC (ug/m3)', 'OA (ug/m3)', 
    'NH4 (ug/m3)', 'NO3 (ug/m3)', 'SO4 (ug/m3)', 'Chi'
])

# Save the current working directory
current_dir = os.getcwd()

# Iterate through 1,000 scenarios (depend on your scenarios)
for k in range(1, 1001):
    folder = f'scenario_{str(k).zfill(4)}'
    output_dir = os.path.join('./', folder, 'out')
    
    # Check if the 'out' directory exists for the current scenario
    if os.path.exists(output_dir):
        os.chdir(output_dir)
        
        # Count the number of .nc files in the current directory
        nc_file_count = count_nc_files('./')
        rows_to_append = []
        
        # Loop through each .nc file in the scenario
        for j in range(1, nc_file_count + 1):
            filename = f'urban_plume_00{str(1).zfill(2)}_000000{str(j).zfill(2)}.nc'
            
            # Load the PartMC data
            pmc = pmcpy.load_pmc(filename)
            
            # Extract gas-phase mixing ratios
            O3 = pmc.get_gas_mixing_ratio(['O3'])
            CO = pmc.get_gas_mixing_ratio(['CO'])
            NO = pmc.get_gas_mixing_ratio(['NO'])
            NOx_list = ['NO', 'NO2']
            NOx = np.sum(pmc.get_gas_mixing_ratio(NOx_list))
            CH3OH = pmc.get_gas_mixing_ratio(['CH3OH'])
            C2H6 = pmc.get_gas_mixing_ratio(['C2H6'])
            ETH = pmc.get_gas_mixing_ratio(['ETH'])
            TOL = pmc.get_gas_mixing_ratio(['TOL'])
            XYL = pmc.get_gas_mixing_ratio(['XYL'])
            ALD2 = pmc.get_gas_mixing_ratio(['ALD2'])
            AONE = pmc.get_gas_mixing_ratio(['AONE'])
            PAR = pmc.get_gas_mixing_ratio(['PAR'])
            OLET = pmc.get_gas_mixing_ratio(['OLET'])
            
            # Define aerosol mass concentration and mixing state groups
            OA_list = ['ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'OC']
            group_list = [['BC'], ['NH4'], ['SO4'], ['NO3'], OA_list]
            
            try:
                # Filter aerosol particles within 100-700 nm and extract relevant properties
                particle_cond = (pmc.get_aero_particle_diameter() <= 7e-7) & (pmc.get_aero_particle_diameter() >= 1e-7)
                BC = pmc.get_aero_mass_conc(['BC'], part_cond=particle_cond) * 1E+9
                BC = BC[0] if np.ndim(BC) > 0 else None
                NH4 = pmc.get_aero_mass_conc(['NH4'], part_cond=particle_cond) * 1E+9
                NH4 = NH4[0] if np.ndim(NH4) > 0 else None
                SO4 = pmc.get_aero_mass_conc(['SO4'], part_cond=particle_cond) * 1E+9
                SO4 = SO4[0] if np.ndim(SO4) > 0 else None
                NO3 = pmc.get_aero_mass_conc(['NO3'], part_cond=particle_cond) * 1E+9
                NO3 = NO3[0] if np.ndim(NO3) > 0 else None
                OA = np.sum(pmc.get_aero_mass_conc(OA_list, part_cond=particle_cond)) * 1E+9
                mixing_state = pmc.get_mixing_state_index(group_list=group_list, part_cond=particle_cond, diversity=False)
            except KeyError:
                BC = NH4 = SO4 = NO3 = OA = mixing_state = 0
            
            # Read environmental data from the NetCDF file
            nc_file = xr.open_dataset(filename, engine='netcdf4')
            Time = nc_file['time']
            RH = nc_file['relative_humidity']
            Dayofyear = nc_file['start_day_of_year']
            temperature_data = nc_file['temperature']
            nc_file.close()
            
            # Append the results for the current file
            rows_to_append.append({
                "Scenario_ID": k,
                "DayofYear": Dayofyear.values,
                "Time(hr)": Time.values / 3600,
                "O3 (ppb)": O3.values[0] if np.ndim(O3.values) > 0 else None,
                "CO (ppb)": CO.values[0] if np.ndim(CO.values) > 0 else None,
                "NO (ppb)": NO.values[0] if np.ndim(NO.values) > 0 else None,
                "NOx (ppb)": NOx.values,
                "CH3OH (ppb)": CH3OH.values[0] if np.ndim(CH3OH.values) > 0 else None,
                "C2H6 (ppb)": C2H6.values[0] if np.ndim(C2H6.values) > 0 else None,
                "ETH (ppb)": ETH.values[0] if np.ndim(ETH.values) > 0 else None,
                "TOL(ppb)": TOL.values[0] if np.ndim(TOL.values) > 0 else None,
                "XYL (ppb)": XYL.values[0] if np.ndim(XYL.values) > 0 else None,
                "ALD2 (ppb)": ALD2.values[0] if np.ndim(ALD2.values) > 0 else None,
                "AONE (ppb)": AONE.values[0] if np.ndim(AONE.values) > 0 else None,
                "PAR (ppb)": PAR.values[0] if np.ndim(PAR.values) > 0 else None,
                "OLET (ppb)": OLET.values[0] if np.ndim(OLET.values) > 0 else None,
                "Temperature(K)": temperature_data.values,
                "RH": RH.values,
                "BC (ug/m3)": BC,
                "OA (ug/m3)": OA,
                "NH4 (ug/m3)": NH4,
                "NO3 (ug/m3)": NO3,
                "SO4 (ug/m3)": SO4,
                "Chi": mixing_state
            })
        
        # Append the collected data to the main DataFrame
        df = pd.concat([df, pd.DataFrame(rows_to_append)], ignore_index=True)
        
        # Return to the original directory
        os.chdir(current_dir)
    else:
        # Skip scenarios without 'out' directories
        continue

# Save the compiled results to a CSV file
output_file = 'PartMC_data_100To700nm.csv'
df.to_csv(output_file, index=False)

print(df)
print(f"Data successfully saved to {output_file}")