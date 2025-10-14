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
drug_regimen = '5-FU'      #regimen:5-FU\Carbo\CDDP\ETO\Carbo+A\CDDP+A|optional:A\VIN

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
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    '0': [0.0000, 0.0000, -0.0001, 0.0000, 0.0000, 0.0000, -0.0003, -0.0047, 0.0001],
    '2': [0.5524, 0.2045, 0.3893, 0.4114, 0.4962, 0.1526, 0.4939, 0.5510, 0.4607],
    '5': [0.6450, 0.3681, 0.4874, 0.5230, 0.5575, 0.3389, 0.4712, 0.6401, 0.5451],
    '10': [0.5229, 0.6269, 0.3638, 0.4239, 0.5592, 0.5743, 0.5093, 0.7002, 0.6707],
    '25': [0.6932, 0.9009, 0.4597, 0.6175, 0.7025, 0.7083, 0.5870, 0.6843, 0.9043],
    '50': [0.7948, 0.9719, 0.7417, 0.6323, 0.7831, 0.7527, 0.7126, 0.7696, 0.9761],
    '100': [0.9471, 0.9939, 0.8779, 0.6838, 0.8829, 0.8225, 0.8959, 0.9336, 0.9900],
    '250': [0.9930, 0.9975, 0.9123, 0.7237, 0.9639, 0.8999, 0.9368, 0.9822, 0.9922],
})

df_group2 = pd.DataFrame({
    'ID': [10, 11, 12, 13, 14, 15, 16],
    '0': [0.0000, 0.0177, 0.0000, 0.0000, 0.0000, 0.0000, None],
    '2': [0.0244, 0.3066, 0.0073, 0.0000, 0.1279, None, 0.7921],
    '10': [0.1705, 0.4607, 0.2827, 0.2837, 0.0576, 0.5517, 0.7807],
    '50': [0.7842, 0.4884, 0.0877, 0.4834, 0.7225, 0.7985, 0.9577],
    '250': [0.9701, 0.8971, 0.1863, 0.7707, 0.9749, 0.9911, 0.9960],
})

df_group3 = pd.DataFrame({
    'ID': [17, 18],
    '0': [-0.1751, 0.0013],
    '4': [0.4357, 0.4146],
    '20': [0.9397, 0.9544],
    '100': [0.9960, 0.9929],
    '500': [0.9935, 0.9946],
})

# Data cleaning
df1 = clean_data(df_group1)
df2 = clean_data(df_group2)
df3 = clean_data(df_group3)

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

# Merge data
def melt_df(df):
    return df.melt(id_vars='ID', var_name='Dose', value_name='Response')

full_data = pd.concat([melt_df(df1), melt_df(df2), melt_df(df3)])
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