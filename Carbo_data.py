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
drug_regimen = 'Carbo'      #regimen:5-FU\Carbo\CDDP\ETO\Carbo+A\CDDP+A|optional:A\VIN

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
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    '0': [0.0000, 0.0000, -0.0001, 0.0000, -0.0003, -0.0047, 0.0001, 0.0000, -0.0002, 0.0000, 0.0000],
    '4': [0.7255, 0.1245, 0.4202, 0.2662, 0.3857, 0.4023, 0.3390, 0.0122, -0.1356, 0.0122, -0.1353],
    '10': [0.8462, 0.1648, 0.4688, 0.2584, 0.4815, 0.4643, 0.8811, 0.9324, -0.2850, 0.9324, -0.2820],
    '20': [0.9908, 0.4501, 0.8708, 0.8751, 0.8973, 0.9643, 0.9938, 0.9624, 0.4804, 0.9623, 0.4749],
    '40': [0.9982, 0.9588, 0.9949, 0.9758, 0.9952, 0.9956, 0.9938, 0.9624, 0.8571, 0.9623, 0.8567],
    '80': [0.9977, 0.9938, 0.9947, 0.9750, 0.9941, 0.9932, 0.9950, 0.9535, 0.9919, 0.9535, 0.9918],
    '160': [0.9994, 0.9974, 0.9967, 0.9828, 0.9954, 0.9927, 0.9946, 0.9593, 0.9919, 0.9595, 0.9918],
    '320': [0.9992, 0.9994, 0.9965, 0.9867, 0.9963, 0.9898, 0.9955, 0.9739, 0.9922, 0.9740, 0.9920]
})

df_group2 = pd.DataFrame({
    'ID': [12],
    '0': [0.0000],
    '1.6': [0.0479],
    '8': [0.4657],
    '40': [0.9396],
    '200': [0.9833]
})

df_group3 = pd.DataFrame({
    'ID': [13],
    '0': [-0.1751],
    '4': [-0.1819],
    '16': [0.3201],
    '80': [0.9839],
    '400': [0.9932]
})

df_group4 = pd.DataFrame({
    'ID': [14, 15],
    '0': [0.001285, 0.017677],
    '3.2': [0.3275195, 0.769931],
    '16': [0.3660645, 0.8229885],
    '80': [0.9932025, 0.995426],
    '400': [0.992174, 0.9956545]
})

df_group5 = pd.DataFrame({
    'ID': [16],
    '0': [0.0000],
    '16': [0.2300],
    '80': [0.9971],
    '200': [0.9988],
    '400': [0.9976]
})

df_group6 = pd.DataFrame({
    'ID': [17],
    '0': [0.0000],
    '16': [0.4299],
    '64': [0.5987],
    '160': [0.7804],
    '320': [0.8321]
})

df_group7 = pd.DataFrame({
    'ID': [18, 19, 20],
    '0': [0.0000, 0.0000, None],
    '3.2': [0.1514, 0.8696, 0.8070],
    '16': [0.3490, 0.9869, 0.8413],
    '80': [0.9953, 0.9994, 0.9988],
    '320': [0.9976, 0.9997, 0.9995],
})

# Data cleaning
df1 = clean_data(df_group1)
df2 = clean_data(df_group2)
df3 = clean_data(df_group3)
df4 = clean_data(df_group4)
df5 = clean_data(df_group5)
df6 = clean_data(df_group6)
df7 = clean_data(df_group7)


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
df6 = impute_missing(df6)
df7 = impute_missing(df7)


# Merge data
def melt_df(df):
    return df.melt(id_vars='ID', var_name='Dose', value_name='Response')


full_data = pd.concat([
    melt_df(df1), melt_df(df2), melt_df(df3),
    melt_df(df4), melt_df(df5), melt_df(df6),
    melt_df(df7)
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