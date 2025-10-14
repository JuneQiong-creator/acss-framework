import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must set before importing pyplot
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# Set the drug regimen
drug_regimen = 'ETO'      #regimen:5-FU\Carbo\CDDP\ETO\Carbo+A\CDDP+A|optional:A\VIN

# Set the working directory
os.chdir(f"C:\\Users\\LIU Qiong\\Desktop\\PhD\\research\\present\\article no.1\\code\\{drug_regimen}")

# ======================
# Data Preprocessing
# ======================
def clean_data(df):
    """Handle negative values and missing values"""
    df = df.copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = np.clip(df[col], 0, 1.2)  # Allow maximum 120% response (consider experimental error)
    return df


# Load data
df_group1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    '0': [-0.1063, 0.0013, 0.0000, 0.0000, -0.1751, None],
    '2': [0.0046, 0.3548, 0.0542, 0.8939, -0.0434, 0.0443],
    '10': [0.0810, 0.4547, 0.0289, 0.8207, 0.1314, 0.2715],
    '50': [0.2299, 0.5481, 0.2582, 0.8864, 0.4105, 0.0286],
    '250': [0.2362, 0.6845, 0.2337, 0.9553, 0.7609, 0.9815]
})

df_group2 = pd.DataFrame({
    'ID': [7, 8],
    '0': [0.0177, 0.0000],
    '4': [0.3323, 0.7480],
    '20': [0.8019, 0.9982],
    '100': [0.8793, 0.9989],
    '500': [0.9399, 0.9983]
})

df_group3 = pd.DataFrame({
    'ID': [9],
    '0': [0.0000],
    '8': [0.1862],
    '10': [0.2803],
    '200': [None],
    '1000': [0.8989]
})

df_group4 = pd.DataFrame({
    'ID': [10],
    '0': [0.0000],
    '2': [0.0600],
    '10': [0.0869],
    '50': [0.1909],
    '250': [0.4773]
})

df_group5 = pd.DataFrame({
    'ID': [11, 12, 13, 14, 15],
    '0': [0.0000, 0.0000, -0.0001, None, None],
    '2': [-0.0986, 0.0927, 0.0219, None, -0.0058],
    '4': [0.1489, -0.0948, -0.0340, None, -0.0064],
    '10': [None, 0.1089, -0.1445, 0.0784, 0.0070],
    '20': [0.3594, None, 0.3638, 0.0909, 0.0482],
    '50': [0.3402, 0.2604, 0.3306, 0.0512, 0.2861],
    '100': [0.2835, 0.3561, 0.5971, None, 0.1521],
    '500': [0.4339, None, 0.7567, 0.0821, 0.2892]
})

# Data cleaning
df1 = clean_data(df_group1)
df2 = clean_data(df_group2)
df3 = clean_data(df_group3)
df4 = clean_data(df_group4)
df5 = clean_data(df_group5)


# Handle missing values
def impute_missing(df):
    """Conservatively impute missing values"""
    df = df.copy()
    # Enforce baseline to be 0
    if '0' in df.columns:
        df['0'] = df['0'].fillna(0)

    # Fill in the missing middle values with the average of adjacent doses
    for i in range(1, len(df.columns) - 1):
        col = df.columns[i]
        if df[col].isna().any():
            prev_col = df.columns[i - 1]
            next_col = df.columns[i + 1]
            df[col] = df[col].fillna((df[prev_col] + df[next_col]) / 2)
    return df


df1 = impute_missing(df1)
df2 = impute_missing(df2)
df3 = impute_missing(df3)
df4 = impute_missing(df4)
df5 = impute_missing(df5)


# Merge data
def melt_df(df):
    return df.melt(id_vars='ID', var_name='Dose', value_name='Response')


full_data = pd.concat([
    melt_df(df1), melt_df(df2), melt_df(df3),
    melt_df(df4), melt_df(df5)
])
full_data['Dose'] = full_data['Dose'].astype(float)
full_data = full_data.dropna().sort_values(['ID', 'Dose'])


def baseline_correction(data):
    """Set zero dose response to zero"""
    corrected = data.copy()
    zero_dose_mask = corrected['Dose'] == 0
    corrected.loc[zero_dose_mask, 'Response'] = 0
    # Verify correction effect
    if not np.allclose(corrected[zero_dose_mask]['Response'], 0):
        print("Warning: Baseline correction did not completely take effect")
    return corrected


full_data = baseline_correction(full_data)
print("===== Data Preprocessing Completed =====")
print(f"Total data points: {len(full_data)}")
print(f"Number of patients: {full_data['ID'].nunique()}")
print("Example data:")
print(full_data.head())

# Save data as CSV
# Rename ID column to Patient_ID
full_data = full_data.rename(columns={'ID': 'Patient_ID'})
# Select necessary columns (ensure Patient_ID, Dose, and Response are included)
output_data = full_data[['Patient_ID', 'Dose', 'Response']].copy()

# Check data quality
print("\n===== Data Quality Check =====")
print(f"Total data points: {len(output_data)}")
print(f"Number of patients: {output_data['Patient_ID'].nunique()}")
print("Dose range:", (output_data['Dose'].min(), output_data['Dose'].max()))
print("Response range:", (output_data['Response'].min(), output_data['Response'].max()))

# Save as CSV (all dose-response data for each patient)
output_path = f"{drug_regimen}.csv"
output_data.to_csv(output_path, index=False)
print(f"\n===== Data Saved =====")
print(f"File Path: {output_path}")
print("Example file content:")
print(output_data.head())

# Validate saved result
try:
    saved_data = pd.read_csv(output_path)
    print("\n===== Validate Saved Result =====")
    print(f"Rows read: {len(saved_data)}")
    print(f"Number of patients: {saved_data['Patient_ID'].nunique()}")
    print("Top 5 Patient IDs:", saved_data['Patient_ID'].unique()[:5])
except Exception as e:
    print(f"Validation failed: {e}")

# Data visualization
# Set styles
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

# Plot all patients overlay visualization
def plot_all_patients(full_data):
    """Display dose-response curves for all patients in one graph"""
    plt.figure(figsize=(14, 8))
    # Assign a unique color to each patient
    unique_ids = full_data['Patient_ID'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
    for pid, color in zip(unique_ids, colors):
        sub = full_data[full_data['Patient_ID'] == pid].sort_values('Dose')
        # Plot the curve (increased line width and marker size)
        plt.plot(sub['Dose'], sub['Response'],
                 'o-', color=color, markersize=6,
                 linewidth=1.5, alpha=0.7,
                 label=f'Patient {pid}')

    # Axis and grid settings
    plt.xlim(-5, full_data['Dose'].max() * 1.1)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Dose (μM)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.title(f'All Patients Dose-Response Raw Curves for {drug_regimen}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Optimize legend display
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               ncol=2, fontsize=8, framealpha=0.5)

    # Annotate key regions
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axvline(0, color='gray', linestyle=':', linewidth=1)
    plt.plot(0, 0, 'ko', markersize=5, label='Origin (0,0)')
    plt.tight_layout()
    plt.savefig(f"{drug_regimen}_all_patients_raw_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

# Execute visualization
if __name__ == "__main__":
    plot_all_patients(full_data)