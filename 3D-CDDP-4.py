import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, bisect
import matplotlib
matplotlib.use('Agg')  # Must set before importing pyplot
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from itertools import combinations


# Set the working directory
os.chdir(r"C:\Users\LIU Qiong\Desktop\PhD\research\present\article no.1\code\CDDP")

np.random.seed(42)

# ======================
# Step 1. Data Preprocessing
# ======================
def clean_data(df):
    """Handle negative values and missing values"""
    df = df.copy()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = np.clip(df[col], 0, 1)
    return df

# Load data
df_group1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    '0': [-0.0003, -0.0047, 0.0001, 0.0000, -0.0186],
    '2': [0.4785, 0.3917, 0.1500, -0.2424, 0.1316],
    '5': [0.8676, 0.6729, 0.2283, -0.1169, 0.0208],
    '10': [0.9815, 0.9848, 0.3586, -0.0518, 0.0426],
    '25': [0.9825, 0.9938, 0.7914, 0.2878, 0.2470],
    '50': [0.9812, 0.9905, 0.9920, 0.5791, 0.4759],
    '100': [0.9867, 0.9947, 0.9922, 0.5787, 0.9539],
    '200': [0.9826, 0.9944, 0.9943, 0.5836, 0.9813]
})

df_group2 = pd.DataFrame({
    'ID': [6, 7, 8, 9, 10, 11],
    '0': [-0.1751, 0.0013, 0.0177, 0.0000, -0.1751, 0.0013],
    '2': [-0.2223, 0.3193, 0.2814, None, -0.2223, 0.2781],
    '10': [0.2765, 0.3884, 0.5462, 0.3232, 0.2765, 0.3801],
    '50': [0.9700, 0.9607, 0.7332, 0.8936, 0.9700, 0.8582],
    '250': [0.9943, 0.9947, 0.9879, 0.9947, 0.9943, 0.8580]
})

df_group3 = pd.DataFrame({
    'ID': [12],
    '0': [0.0000],
    '8': [0.3262],
    '40': [0.6206],
    '100': [0.7488],
    '200': [0.9437]
})

df_group4 = pd.DataFrame({
    'ID': [13, 14, 15, 16],
    '0': [0.0000, 0.0000, None, 0.0000],
    '1.6': [0.2508, 0.1172, 0.4442, -0.0861],
    '8': [0.2039, 0.8413, 0.7774, 0.0186],
    '40': [0.4787, 0.9416, 0.8409, 0.0823],
    '200': [0.9877, 0.9937, 0.9942, 0.7568]
})

# Data cleaning
df1 = clean_data(df_group1)
df2 = clean_data(df_group2)
df3 = clean_data(df_group3)
df4 = clean_data(df_group4)

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

# Merge data
def melt_df(df):
    return df.melt(id_vars='ID', var_name='Dose', value_name='Response')

full_data = pd.concat([
    melt_df(df1), melt_df(df2), melt_df(df3),
    melt_df(df4)
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
output_path = "CDDP_dose_response_data.csv"
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
    plt.title('All Patients Dose-Response Curves for CDDP', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Optimize legend display
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               ncol=2, fontsize=8, framealpha=0.5)

    # Annotate key regions
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axvline(0, color='gray', linestyle=':', linewidth=1)
    plt.plot(0, 0, 'ko', markersize=5, label='Origin (0,0)')
    plt.tight_layout()
    plt.savefig("CDDP_all_patients_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# Execute visualization
if __name__ == "__main__":
    plot_all_patients(full_data)





# ======================
# Step 2. Model Selection
# ======================
def model_selection(full_data):
    results = {}

    # Define candidate models
    def sigEmax(x, ec50, hill, e0, emax):
        return e0 + emax * (x ** hill) / (ec50 ** hill + x ** hill)

    def Emax(x, ec50, e0, emax):
        return e0 + emax * x / (ec50 + x)

    def Quadratic(x, beta1, beta2, e0):
        return e0 + beta1 * x + beta2 * (x ** 2)

    def Logistic(x, ec50, delta, e0, emax):
        return e0 + emax / (1 + np.exp((ec50 - x) / delta))

    models = {
        'sigEmax': sigEmax,
        'Emax': Emax,
        'Quadratic': Quadratic,
        'Logistic': Logistic
    }

    def select_model(patient_results):
        best_by_aic = min(patient_results.items(), key=lambda x: x[1]['aic'])
        aic_min = best_by_aic[1]['aic']
        for name, metrics in patient_results.items():
            if (metrics['aic'] - aic_min < 2) and (metrics['r2'] > best_by_aic[1]['r2'] + 0.05):
                print(
                    f"Patient {pid}: Model {name} deserves consideration (ΔAIC={metrics['aic'] - aic_min:.1f}, ΔR²={metrics['r2'] - best_by_aic[1]['r2']:.2f})")
        return best_by_aic

    for pid in full_data['Patient_ID'].unique():
        sub = full_data[full_data['Patient_ID'] == pid]
        x = sub['Dose'].values
        y = sub['Response'].values
        patient_results = {}

        for name, func in models.items():
            try:
                if name == 'sigEmax':
                    p0 = [np.median(x), 1.5, 0, 1.0]
                    bounds = ([0.1, 0.3, 0, 0.1], [max(x) * 10, 5, 0.1, 1.5])
                elif name == 'Emax':
                    p0 = [np.median(x), 0, 1.0]
                    bounds = ([0.1, 0, 0.1], [max(x) * 10, 0.1, 1.5])
                elif name == 'Logistic':
                    p0 = [np.median(x), 1.0, 0, 1.0]
                    bounds = ([0.1, 0.1, 0, 0.1], [max(x) * 10, 10.0, 0.1, 1.5])
                else:  # Quadratic
                    p0 = [0.1, 0.01, 0]  # β1, β2, E0
                    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 0.1])

                params, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, method='trf', maxfev=10000)
                y_pred = func(x, *params)

                # Calculate evaluation metrics
                ssr = np.sum((y - y_pred) ** 2)
                n = len(y)
                k = len(params)
                patient_results[name] = {
                    'params': params,
                    'r2': r2_score(y, y_pred),
                    'aic': 2 * k + n * np.log(ssr / n),
                    'bic': k * np.log(n) + n * np.log(ssr / n),
                    'rmse': np.sqrt(ssr / n),
                    'delta_qua': params[1] / params[0] if name == 'Quadratic' else None  # Compute δ=β2/β1
                }
            except Exception as e:
                print(f"Patient {pid} {name} fit failed: {str(e)}")
                continue

        if patient_results:
            best_model_info = select_model(patient_results)
            results[pid] = {
                'best_model': best_model_info[0],
                'all_models': patient_results,
                'selected_params': best_model_info[1]['params'],
                'delta_qua': best_model_info[1]['delta_qua'] if best_model_info[0] == 'Quadratic' else None
            }
    return results

# Execute and print detailed results
results = model_selection(full_data)
for pid, res in results.items():
    print(f"\n{'=' * 30}")
    print(f"Patient {pid} - Best Model: {res['best_model']}")
    print(f"{'=' * 30}")

    # Print model comparison
    print("\nModel Comparison:")
    print(f"{'Model':<10} {'AIC':<8} {'BIC':<8} {'R²':<6} {'RMSE':<6}")
    print("-" * 40)
    for name, metrics in res['all_models'].items():
        print(f"{name:<10} {metrics['aic']:<8.1f} {metrics['bic']:<8.1f} "
              f"{metrics['r2']:<6.2f} {metrics['rmse']:<6.4f}")

    # Print best parameters
    print("\nBest Parameters:")
    if res['best_model'] == 'sigEmax':
        ec50, hill, e0, emax = res['selected_params']
        print(f"EC50={ec50:.1f}μM, Hill={hill:.1f}, E0={e0:.2f}, Emax={emax:.2f}")
    elif res['best_model'] == 'Emax':
        ec50, e0, emax = res['selected_params']
        print(f"EC50={ec50:.1f}μM, E0={e0:.2f}, Emax={emax:.2f}")
    elif res['best_model'] == 'Logistic':
        ec50, delta, e0, emax = res['selected_params']
        print(f"EC50={ec50:.1f}μM, δ={delta:.1f}, E0={e0:.2f}, Emax={emax:.2f}")
    else:  # Quadratic
        beta1, beta2, e0 = res['selected_params']
        print(f"β1={beta1:.4f}, β2={beta2:.4f}, E0={e0:.2f}")
        print(f"Coefficient ratio δ=β2/β1 = {res['delta_qua']:.4f}")

# Visualization of fitting results
def plot_best_fit(pid, full_data, results):
    sub = full_data[full_data['Patient_ID'] == pid].sort_values('Dose')
    x = sub['Dose'].values
    y = sub['Response'].values
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=100, color='steelblue', label='Observed Data')
    x_fit = np.linspace(0, max(x) * 1.1, 100)
    model_info = results[pid]

    if model_info['best_model'] == 'Emax':
        ec50, e0, emax = model_info['selected_params']
        y_fit = e0 + emax * x_fit / (ec50 + x_fit)
        equation = f"$y = {e0:.2f} + {emax:.2f}x/({ec50:.1f} + x)$"
    elif model_info['best_model'] == 'sigEmax':
        ec50, hill, e0, emax = model_info['selected_params']
        y_fit = e0 + emax * (x_fit ** hill) / (ec50 ** hill + x_fit ** hill)
        equation = f"$y = {e0:.2f} + {emax:.2f}·x^{{{hill:.1f}}}/({ec50:.1f}^{{{hill:.1f}}} + x^{{{hill:.1f}}})$"
    elif model_info['best_model'] == 'Logistic':
        ec50, delta, e0, emax = model_info['selected_params']
        y_fit = e0 + emax / (1 + np.exp((ec50 - x_fit) / delta))
        equation = f"$y = {e0:.2f} + {emax:.2f}/(1+e^{{({ec50:.1f}-x)/{delta:.1f}}})$"
    else:  # Quadratic
        beta1, beta2, e0 = model_info['selected_params']
        y_fit = e0 + beta1 * x_fit + beta2 * (x_fit ** 2)
        equation = f"$y = {e0:.2f} + {beta1:.3f}x + {beta2:.3f}x^2$\n$\delta=\\beta_2/\\beta_1={model_info['delta_qua']:.2f}$"

    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Best Fit ({model_info["best_model"]})')
    plt.annotate(equation, xy=(0.5, 0.2), xycoords='axes fraction',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.xlabel('Dose (μM)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.title(f'Patient {pid} Dose-Response Curve\nBest Model: {model_info["best_model"]}', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join("dose_response_plots", f"patient_{pid}_dose_response.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Generate fit plots for all patients
os.makedirs("dose_response_plots", exist_ok=True)
for pid in results.keys():
    plot_best_fit(pid, full_data, results)

# Save fitting results
def save_results(results, full_data):
    csv_data = []
    for pid, res in results.items():
        row = {
            'Patient_ID': pid,
            'Best_Model': res['best_model'],
            'Dose_Range': f"{full_data[full_data['Patient_ID'] == pid]['Dose'].min():.1f}-{full_data[full_data['Patient_ID'] == pid]['Dose'].max():.1f}μM",
            'Data_Points': len(full_data[full_data['Patient_ID'] == pid]),
            'Quadratic_Delta': res['delta_qua'] if res['best_model'] == 'Quadratic' else None
        }

        # Model comparison metrics
        for model_name in ['sigEmax', 'Emax', 'Quadratic', 'Logistic']:
            if model_name in res['all_models']:
                metrics = res['all_models'][model_name]
                row.update({
                    f"{model_name}_AIC": metrics['aic'],
                    f"{model_name}_BIC": metrics['bic'],
                    f"{model_name}_R2": metrics['r2'],
                    f"{model_name}_RMSE": metrics['rmse'],
                    f"{model_name}_Quadratic_Delta": metrics['delta_qua'] if model_name == 'Quadratic' else None
                })
            else:
                row.update({
                    f"{model_name}_AIC": None,
                    f"{model_name}_BIC": None,
                    f"{model_name}_R2": None,
                    f"{model_name}_RMSE": None,
                    f"{model_name}_Quadratic_Delta": None
                })

        # Best model parameters
        params = res['selected_params']
        if res['best_model'] == 'sigEmax':
            row.update({
                'EC50': params[0], 'Hill': params[1], 'E0': params[2], 'Emax': params[3],
                'beta1': None, 'beta2': None
            })
        elif res['best_model'] == 'Emax':
            row.update({
                'EC50': params[0], 'E0': params[1], 'Emax': params[2],
                'Hill': None, 'beta1': None, 'beta2': None
            })
        elif res['best_model'] == 'Logistic':
            row.update({
                'EC50': params[0], 'Delta_logistic': params[1], 'E0': params[2], 'Emax': params[3],
                'Hill': None, 'beta1': None, 'beta2': None
            })
        else:  # Quadratic
            row.update({
                'beta1': params[0], 'beta2': params[1], 'E0': params[2],
                'EC50': None, 'Hill': None, 'Emax': None
            })

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    columns_order = [
        'Patient_ID', 'Best_Model', 'Data_Points', 'Dose_Range',
        'sigEmax_AIC', 'sigEmax_BIC', 'sigEmax_R2', 'sigEmax_RMSE',
        'Emax_AIC', 'Emax_BIC', 'Emax_R2', 'Emax_RMSE',
        'Quadratic_AIC', 'Quadratic_BIC', 'Quadratic_R2', 'Quadratic_RMSE', 'Quadratic_Delta',
        'Logistic_AIC', 'Logistic_BIC', 'Logistic_R2', 'Logistic_RMSE',
        'EC50', 'Hill', 'Delta_logistic', 'E0', 'Emax', 'beta1', 'beta2'
    ]
    df = df[columns_order]
    df.to_csv("model_selection_results.csv", index=False, float_format="%.4f")

save_results(results, full_data)

def calculate_best_model_stats(results):
    """Calculate the statistical information on which models become the best models"""
    # Count the number of times each model was selected as the best model
    model_counts = {}
    for pid, res in results.items():
        model = res['best_model']
        model_counts[model] = model_counts.get(model, 0) + 1

    # Calculate total numbers and percentages
    total_patients = len(results)
    model_stats = []
    for model, count in model_counts.items():
        model_stats.append({
            'Model': model,
            'Count': count,
            'Percentage': count / total_patients * 100
        })

    # Convert to DataFrame and sort
    stats_df = pd.DataFrame(model_stats)
    stats_df = stats_df.sort_values('Count', ascending=False)

    return stats_df

# Calculate and display statistical results
model_stats = calculate_best_model_stats(results)
print("\n" + "=" * 50)
print("Statistical results of each model being selected as the best model:")
print("=" * 50)
print(model_stats.to_string(index=False))

# Identify the best fitting model
best_overall_model = model_stats.iloc[0]['Model']
best_percentage = model_stats.iloc[0]['Percentage']
print(
    f"\nConclusion: The best fitting model is {best_overall_model}, which fitted {model_stats.iloc[0]['Count']} times out of {len(results)} patients")
print(f"Fitting percentage: {best_percentage:.1f}%")

# Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(model_stats['Model'], model_stats['Count'],
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Add numerical labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height}\n({height / len(results) * 100:.1f}%)',
             ha='center', va='bottom')

plt.title('Counts of Each Model Being Selected as the Best Model CDDP', fontsize=14, pad=20)
plt.xlabel('Model Type', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.xticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add additional notes
plt.annotate(f'Best Model: {best_overall_model}\nFitting Percentage: {best_percentage:.1f}%',
             xy=(0.7, 0.8), xycoords='axes fraction',
             bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

plt.tight_layout()
plt.savefig("best_model_distribution.png", dpi=300)
plt.show()

# Save statistical results to CSV
model_stats.to_csv("best_model_statistics.csv", index=False)
print("\nStatistical results saved to best_model_statistics.csv")







# ======================
# Step 3. Optimal Model Fitting + Fill Missing Values
# ======================

def fit_sigmoid_emax_all(full_data):
    sigmoid_emax_results = {}

    def sigmoid_emax(x, ec50, e0, emax, h):
        """Sigmoid Emax model equation"""
        return e0 + emax * (x ** h) / (ec50 ** h + x ** h)

    def sigmoid_emax_derivative(x, ec50, e0, emax, h):
        """First derivative (used to calculate the slope at EC50)"""
        numerator = (h * x ** (h - 1)) * (ec50 ** h + x ** h) - (h * x ** (2 * h - 1))
        denominator = (ec50 ** h + x ** h) ** 2
        return emax * (numerator / denominator)

    for pid in full_data['Patient_ID'].unique():
        sub = full_data[full_data['Patient_ID'] == pid]
        x = sub['Dose'].values
        y = sub['Response'].values

        try:
            # Initial parameter estimates
            p0 = [
                np.median(x[x > 0]),  # Initial EC50 as median dose
                max(0, y[x == 0].mean() if 0 in x else np.percentile(y, 10)),  # E0
                np.percentile(y, 90),  # Initial Emax using the 90th percentile
                1.5  # Initial Hill coefficient
            ]
            # Parameter bounds
            bounds = (
                [0.01, 0, 0, 0.1],  # Lower bounds
                [max(x) * 10, 0.1, 1, 10.0]  # Upper bounds
            )
            # Curve fitting
            params, pcov = curve_fit(sigmoid_emax, x, y, p0=p0, bounds=bounds, maxfev=5000)
            ec50, e0, emax_val, h = params

            # Calculate MED (dose at Response=0.2)
            def med_equation(dose):
                return sigmoid_emax(dose, *params) - 0.2

            try:
                med = bisect(med_equation, 0, max(x) * 10, xtol=0.01) if emax_val > 0.2 else np.nan
            except:
                med = np.nan

            # Calculate IC50 (dose at Response=50%)
            def ic50_equation(dose):
                target_response = 0.5
                return sigmoid_emax(dose, *params) - target_response

            try:
                ic50 = bisect(ic50_equation, 0, max(x) * 10, xtol=0.01) if (emax_val - e0) > 0.1 else np.nan
            except:
                ic50 = np.nan

            # Calculate IC90 (dose at Response=90%)
            def ic90_equation(dose):
                target_response = 0.9
                return sigmoid_emax(dose, *params) - target_response

            try:
                ic90 = bisect(ic90_equation, 0, max(x) * 10, xtol=0.01) if (emax_val - e0) > 0.1 else np.nan
            except:
                ic90 = np.nan

            # Calculate the slope at EC50
            slope_at_ec50 = sigmoid_emax_derivative(ec50, *params)

            # Calculate fitting metrics
            y_pred = sigmoid_emax(x, *params)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            sigmoid_emax_results[pid] = {
                'EC50': ec50,
                'E0': e0,
                'Emax': emax_val,
                'Hill': h,
                'R2': r2,
                'RMSE': rmse,
                'MED': med,
                'IC50': ic50,
                'IC90': ic90,
                'Slope_at_EC50': slope_at_ec50,
                'Fit_Success': True,
                'params': params  # Save parameters for subsequent predictions
            }
        except Exception as e:
            print(f"Patient {pid} Sigmoid Emax fit failed: {str(e)}")
            sigmoid_emax_results[pid] = None

    return sigmoid_emax_results


def prepare_sigmoid_results_filled(sigmoid_emax_results):
    """
    Prepare results DataFrame from sigmoid Emax fitting results.
    Fill missing MED/IC50/IC90 with the max of that column across all patients.
    Mark which values are filled using 'Filled_FLAG'.
    """
    results_list = []

    # First pass: collect all computed values to find max
    med_list, ic50_list, ic90_list = [], [], []
    for pid, res in sigmoid_emax_results.items():
        if res is not None:
            med_list.append(res['MED'])
            ic50_list.append(res['IC50'])
            ic90_list.append(res['IC90'])
    # Compute max ignoring nan
    med_max = np.nanmax(med_list)
    ic50_max = np.nanmax(ic50_list)
    ic90_max = np.nanmax(ic90_list)

    for pid, res in sigmoid_emax_results.items():
        row = {'Patient_ID': pid, 'Filled_FLAG': False}
        if res is not None:
            row.update({
                'EC50': res['EC50'],
                'E0': res['E0'],
                'Emax': res['Emax'],
                'Hill': res['Hill'],
                'R2': res['R2'],
                'RMSE': res['RMSE'],
                'Slope_at_EC50': res['Slope_at_EC50'],
                'Fit_Success': True
            })
            # Fill missing MED/IC50/IC90 with max if nan
            row['MED'] = res['MED'] if not np.isnan(res['MED']) else med_max
            row['IC50'] = res['IC50'] if not np.isnan(res['IC50']) else ic50_max
            row['IC90'] = res['IC90'] if not np.isnan(res['IC90']) else ic90_max
            # Mark filled
            if np.isnan(res['MED']) or np.isnan(res['IC50']) or np.isnan(res['IC90']):
                row['Filled_FLAG'] = True
        else:
            # Failed fit: fill all with max
            row.update({
                'EC50': None,
                'E0': None,
                'Emax': None,
                'Hill': None,
                'R2': None,
                'RMSE': None,
                'Slope_at_EC50': None,
                'Fit_Success': False,
                'MED': med_max,
                'IC50': ic50_max,
                'IC90': ic90_max,
                'Filled_FLAG': True
            })
        results_list.append(row)
    return pd.DataFrame(results_list)


if __name__ == "__main__":
    sigmoid_results = fit_sigmoid_emax_all(full_data)
    results_df = prepare_sigmoid_results_filled(sigmoid_results)

    # Add dose range information
    dose_ranges = full_data.groupby('Patient_ID')['Dose'].agg(['min', 'max'])
    results_df = results_df.merge(dose_ranges, left_on='Patient_ID', right_index=True)
    results_df.rename(columns={'min': 'Min_Dose', 'max': 'Max_Dose'}, inplace=True)

    # Adjust column order
    column_order = [
        'Patient_ID', 'Fit_Success', 'Filled_FLAG', 'Min_Dose', 'Max_Dose',
        'EC50', 'E0', 'Emax', 'Hill',
        'R2', 'RMSE', 'MED', 'IC50', 'IC90', 'Slope_at_EC50'
    ]
    results_df = results_df[column_order]

    # Format numerical values
    float_cols = ['EC50', 'E0', 'Emax', 'Hill',
                  'R2', 'RMSE', 'MED', 'IC50', 'IC90', 'Slope_at_EC50']
    results_df[float_cols] = results_df[float_cols].round({
        'EC50': 2,
        'E0': 2,
        'Emax': 2,
        'Hill': 2,
        'R2': 4,
        'RMSE': 4,
        'MED': 2,
        'IC50': 2,
        'IC90': 2,
        'Slope_at_EC50': 4
    })

    # Save filled results
    output_file = "fitted_metrics.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Filled MED/IC50/IC90 and saved results to {output_file}")
    print("\nPreview:")
    print(results_df.head())

# Model diagnostic recommendations
def diagnostic_checks(results_df):
    # Check distribution of Hill coefficients
    h_values = results_df['Hill'].dropna()
    print(f"\nHill coefficient statistics:\n{h_values.describe()}")

    # Check fitting success rate
    success_rate = results_df['Fit_Success'].mean()
    print(f"\nFitting success rate: {success_rate:.1%}")

    # Identify poor fits (R² < 0.5)
    poor_fits = results_df[results_df['R2'] < 0.5]
    print(f"\nPatients to check (R² < 0.5):\n{poor_fits[['Patient_ID', 'R2']]}")

# Execute diagnostics
diagnostic_checks(results_df)

# Visualization of all patient fits
plt.figure(figsize=(12, 8))

def sigmoid_emax_func(x, ec50, e0, emax, hill):
    """Sigmoid Emax model function (identical to fitting function)"""
    return e0 + emax * (x ** hill) / (ec50 ** hill + x ** hill)

# Create color mapping
cmap = plt.colormaps['tab20']
colors = [cmap(i) for i in np.linspace(0, 1, len(sigmoid_results))]

# Create a unified dose range
x_fit = np.linspace(0, 250, 200)

# Plot the fitting curves for all patients
for (pid, res), color in zip(sigmoid_results.items(), colors):
    if res is not None and res['Fit_Success']:  # Only plot successfully fitted patients
        # Correctly extract parameters (pay attention to order)
        ec50 = res['EC50']
        e0 = res['E0']
        emax = res['Emax']
        hill = res['Hill']
        y_fit = sigmoid_emax_func(x_fit, ec50, e0, emax, hill)
        plt.plot(x_fit, y_fit,
                 label=f'Patient {pid}\nEC50={ec50:.1f}μM\nHill={hill:.1f}',
                 color=color,
                 linewidth=2,
                 alpha=0.8)
        # Mark original data points
        sub = full_data[full_data['Patient_ID'] == pid]
        plt.scatter(sub['Dose'], sub['Response'],
                    color=color, s=50, edgecolor='white', zorder=3)

# Chart decoration
plt.title('CDDP Dose-Response Curves (Sigmoid Emax Model)', fontsize=14, pad=20)
plt.xlabel('Dose (μM)', fontsize=12)
plt.ylabel('Response', fontsize=12)
plt.xlim(0, 250)
plt.ylim(-0.05, 1.2)
plt.axhline(0, color='gray', linestyle=':', linewidth=2)
plt.axhline(1, color='gray', linestyle=':', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           title='Patient Parameters', fontsize=8, framealpha=0.5, ncol=2)
plt.tight_layout()
plt.savefig('all_patients_best_model_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# New step: Generate fitted 8-point response values (model decides dose=0)
def generate_fitted_responses(sigmoid_results, fixed_doses=[0, 2, 5, 10, 25, 50, 100, 250]):
    """Use sigEmax model to predict fixed 8-point responses without forcing dose=0 to 0"""

    def sigmoid_emax(x, ec50, e0, emax, h):
        return e0 + emax * (x ** h) / (ec50 ** h + x ** h)

    fitted_data = []
    for pid, res in sigmoid_results.items():
        if res is not None and res['Fit_Success']:
            params = res['params']  # [ec50, e0, emax, h]
            row = {'Patient_ID': pid}
            for dose in fixed_doses:
                response = sigmoid_emax(dose, *params)
                response = np.clip(response, 0, 1.2)  # Clip abnormal values
                row[f'Response_{dose}'] = response
            fitted_data.append(row)
        else:
            # Use NaN for failed patients
            row = {'Patient_ID': pid}
            for dose in fixed_doses:
                row[f'Response_{dose}'] = np.nan
            fitted_data.append(row)

    fitted_df = pd.DataFrame(fitted_data)
    fitted_df.to_csv("fitted_data.csv", index=False)
    print("Fitted responses saved to fitted_data.csv")
    print("Preview:")
    print(fitted_df.head())
    return fitted_df

# Execute generation of fitted responses
fitted_responses_df = generate_fitted_responses(sigmoid_results)






# ======================
# Step 4. Pre-Classification
# ======================

# --- Load fitted responses data ---
fitted_df = pd.read_csv("fitted_data.csv")
response_cols = [col for col in fitted_df.columns if col.startswith('Response_')]
fitted_df['Patient_ID'] = fitted_df['Patient_ID'].astype(str)
fitted_df = fitted_df.set_index('Patient_ID')

# Fill missing response values with median
if fitted_df[response_cols].isna().any().any():
    print("Warning: Missing values detected in response columns. Filling with median.")
    fitted_df[response_cols] = fitted_df[response_cols].fillna(fitted_df[response_cols].median())

# --- Define dose points and weights for CRS ---
dose_points = [2, 5, 10, 25, 50, 100, 250]
weights = np.array([0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.10])
weights = weights / weights.sum()

# --- Calculate Composite Response Score (CRS) ---
composite_response = pd.Series(0.0, index=fitted_df.index)
for dose, weight in zip(dose_points, weights):
    composite_response += fitted_df[f'Response_{dose}'] * weight

# Normalize to 0-100
composite_response = 100 * (composite_response - composite_response.min()) / (composite_response.max() - composite_response.min())
composite_response.name = 'CRS'

# --- Load metrics from Step 3 ---
metrics_df = pd.read_csv("fitted_metrics.csv")
metrics_df['Patient_ID'] = metrics_df['Patient_ID'].astype(str)
metrics_df = metrics_df.set_index('Patient_ID')

# Extract MED, IC50, IC90 and merge with CRS
use_metrics_df = metrics_df[['MED','IC50','IC90']].copy()
use_metrics_df = use_metrics_df.join(composite_response)

# Save combined metrics
use_metrics_df.to_csv("use_metrics.csv")
print(f"Saved combined CRS + MED/IC50/IC90 to use_metrics.csv")

# --- Apply pre-classification for missing IC50 ---
#preclassified_mask = use_metrics_df['IC50'].isna()
#preclassified_ids = use_metrics_df.index[preclassified_mask].tolist()

#preclassified_df = use_metrics_df.loc[preclassified_ids].copy()
#preclassified_df['Pre_C'] = 'Low'

#preclassified_df.to_csv("5-FU_preclassified_patients.csv")
#print(f"Pre-classified {len(preclassified_df)} patients with missing IC50 as Low.")
#print(preclassified_df)

# --- Remaining patients (with IC50 available) ---
# no-preclassification
remaining_df = use_metrics_df.copy()





# ======================
# Step 5: Grid Search to Find Optimal k1, k2
# ======================

# 从 Step 4 加载数据
fitted_df = pd.read_csv("fitted_data.csv")
response_cols = [col for col in fitted_df.columns if col.startswith('Response_')]
fitted_df['Patient_ID'] = fitted_df['Patient_ID'].astype(str)
fitted_df = fitted_df.set_index('Patient_ID')

# 填充缺失响应值
if fitted_df[response_cols].isna().any().any():
    print("Warning: Missing values detected in response columns. Filling with median.")
    fitted_df[response_cols] = fitted_df[response_cols].fillna(fitted_df[response_cols].median())

metrics_df = pd.read_csv("fitted_metrics.csv")
metrics_df['Patient_ID'] = metrics_df['Patient_ID'].astype(str)
metrics_df = metrics_df.set_index('Patient_ID')

use_metrics_df = pd.read_csv("use_metrics.csv")
use_metrics_df['Patient_ID'] = use_metrics_df['Patient_ID'].astype(str)
use_metrics_df = use_metrics_df.set_index('Patient_ID')

#preclassified_df = pd.read_csv("5-FU_preclassified_patients.csv")
#preclassified_df['Patient_ID'] = preclassified_df['Patient_ID'].astype(str)
#preclassified_df = preclassified_df.set_index('Patient_ID')
#preclassified_df['Pre_C'] = 'Low'

#remaining_df = pd.read_csv("5-FU_after_preclassification.csv")
remaining_df = pd.read_csv("use_metrics.csv")
remaining_df['Patient_ID'] = remaining_df['Patient_ID'].astype(str)
remaining_df = remaining_df.set_index('Patient_ID')

# 定义曲线特征（基于拟合响应值）
curve_features_df = fitted_df[response_cols].loc[remaining_df.index]

# Helper functions
def calculate_balance_score(groups):
    """计算组平衡分数基于熵"""
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
    return balance_score

def std_based_classification_dynamic_k(scores_series, k1, k2, apply_correction=True):
    """基于均值和std的动态k分类"""
    scores = np.asarray(scores_series)
    mean = np.mean(scores)
    std = np.std(scores)
    if apply_correction and len(scores) < 30 and len(scores) > 1.5:
        std = std * np.sqrt(len(scores) / (len(scores) - 1.5))
    low_thresh = mean - k1 * std
    high_thresh = mean + k2 * std
    groups = np.select(
        [scores <= low_thresh, scores >= high_thresh],
        ['Low', 'High'],
        default='Intermediate'
    )
    return groups, {'mean': mean, 'std': std, 'low_thresh': low_thresh, 'high_thresh': high_thresh}

def perform_loocv_for_k_detailed(scores_series, k1, k2, n_min=1, output_dir="k_LOOCV_Results"):
    """执行LOOCV，计算MCR和LMR，并记录留出患者的分类结果"""
    if not isinstance(scores_series, pd.Series):
        raise ValueError("scores_series must be a pandas Series indexed by Patient_ID")

    Path(output_dir).mkdir(exist_ok=True)

    patient_ids = scores_series.index.to_list()
    n_patients = len(patient_ids)
    if n_patients < 3:
        return np.nan, np.nan, pd.DataFrame()

    original_groups, original_params = std_based_classification_dynamic_k(scores_series, k1, k2)
    original_groups_series = pd.Series(original_groups, index=patient_ids)

    consistency_records = []  # 剩余患者的改变标志
    left_out_match_flags = []  # 留出患者的匹配标志
    loocv_results = []
    skipped_count = 0
    valid_count = 0

    print(f"Starting LOOCV with {len(patient_ids)} patients...")

    for left_out_id in patient_ids:
        loo_scores = scores_series.drop(labels=left_out_id)
        if len(loo_scores) < 3:
            skipped_count += 1
            print(f"Skipped {left_out_id}: Not enough samples ({len(loo_scores)})")
            continue

        loo_groups, loo_params = std_based_classification_dynamic_k(loo_scores, k1, k2)
        loo_groups_series = pd.Series(loo_groups, index=loo_scores.index)

        # Check groups (use fallback if needed)
        unique_loo, counts_loo = np.unique(loo_groups, return_counts=True)
        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            # Fallback: use quantile-based grouping
            quantiles = np.percentile(loo_scores.values, [33, 66])
            loo_groups_fallback = np.where(loo_scores.values < quantiles[0], 'Low',
                                          np.where(loo_scores.values > quantiles[1], 'High', 'Intermediate'))
            loo_groups_series = pd.Series(loo_groups_fallback, index=loo_scores.index)
            unique_loo, counts_loo = np.unique(loo_groups_series, return_counts=True)
            print(f"Fallback used for {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")

        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            skipped_count += 1
            print(f"Skipped {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")
            continue

        valid_count += 1
        print(f"Processed {left_out_id}: groups={unique_loo}, counts={counts_loo}")

        # 计算留出患者的分类
        left_out_score = float(scores_series.loc[left_out_id])
        loo_low = loo_params['low_thresh']
        loo_high = loo_params['high_thresh']
        loo_label_leftout = (
            'Low' if left_out_score <= loo_low else
            'High' if left_out_score >= loo_high else
            'Intermediate'
        )
        orig_label_leftout = original_groups_series.loc[left_out_id]
        left_out_match = int(loo_label_leftout == orig_label_leftout)
        left_out_match_flags.append(left_out_match)

        # 记录剩余患者结果，并添加留出患者的分类信息
        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_scores.index,
            'Composite_Response_Score': loo_scores.values,
            'LOOCV_Group': loo_groups_series.values,
            'Original_Group': original_groups_series.loc[loo_scores.index].values,
            'Mean': loo_params['mean'],
            'Std': loo_params['std'],
            'Low_Threshold': loo_params['low_thresh'],
            'High_Threshold': loo_params['high_thresh'],
            'Left_Out_ID': left_out_id,
            'Left_Out_Score': left_out_score,
            'Left_Out_LOOCV_Group': loo_label_leftout,
            'Left_Out_Original_Group': orig_label_leftout,
            'Left_Out_Match': left_out_match
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_filename = f"{output_dir}/LOOCV_Result_Patient_{left_out_id}.csv"
        loo_result_df.to_csv(loo_filename, index=False)
        loocv_results.append(loo_result_df)

        for pid in loo_scores.index:
            changed_flag = int(original_groups_series.loc[pid] != loo_groups_series.loc[pid])
            consistency_records.append(changed_flag)

    print(f"LOOCV: Valid iterations = {valid_count}, Skipped = {skipped_count}")
    mean_change_rate = float(np.mean(consistency_records)) if consistency_records else np.nan
    left_out_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()

    return mean_change_rate, left_out_match_rate, loocv_results_df

def find_optimal_k_with_css_cbs_crs(scores_series, curve_features_df, results_df,
                                    w_separation=1/3, w_balance=1/3, w_robustness=1/3,
                                    n_min=1, k_min=0.05, k_max=1.5, k_step=0.05,
                                    verbose=True, output_dir="k_LOOCV_Results"):
    """基于CSS, CBS, CRS的网格搜索最优k1, k2"""
    k_range = np.arange(k_min, k_max + 1e-9, k_step)
    search_results = []

    if verbose:
        print(f"\nSearching k1,k2 in [{k_min},{k_max}] step {k_step} ({len(k_range) ** 2} combos)")

    total = len(k_range) * len(k_range)
    cnt = 0
    valid_count = 0

    if results_df.index.name != 'Patient_ID':
        if 'Patient_ID' in results_df.columns:
            results_df = results_df.set_index('Patient_ID')
        else:
            raise KeyError("Patient_ID not found in results_df columns or index")

    param_features = results_df.loc[scores_series.index.intersection(results_df.index), ['MED', 'IC50', 'IC90']].copy()
    param_features = np.log1p(param_features.fillna(param_features.median()))

    for k1 in k_range:
        for k2 in k_range:
            cnt += 1
            if verbose:
                print(f"\r[{cnt}/{total}] k1={k1:.2f}, k2={k2:.2f}...", end="")

            groups, _ = std_based_classification_dynamic_k(scores_series, k1, k2)
            unique_labels, counts = np.unique(groups, return_counts=True)
            if len(unique_labels) != 3 or np.any(counts < n_min):
                continue

            try:
                labels_series = pd.Series(groups, index=scores_series.index)
                labels_codes = labels_series.astype('category').cat.codes

                sil_curve = silhouette_score(curve_features_df.values, labels_codes)
                db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

                cohens_ds = []
                for param in ['MED', 'IC50', 'IC90']:
                    param_values = param_features[param].values
                    group_values = [param_values[labels_codes == i] for i in range(3) if len(param_values[labels_codes == i]) > 1]
                    if len(group_values) == 3:
                        for g1, g2 in combinations(range(3), 2):
                            if len(group_values[g1]) > 1 and len(group_values[g2]) > 1:
                                mean1, mean2 = np.mean(group_values[g1]), np.mean(group_values[g2])
                                std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
                                n1, n2 = len(group_values[g1]), len(group_values[g2])
                                pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
                                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
                                cohens_ds.append(abs(cohens_d))
                effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

            except Exception as e:
                if verbose:
                    print(f"\nCurve metric error for k1={k1}, k2={k2}: {e}")
                continue

            balance = calculate_balance_score(groups)

            mean_change_rate, left_out_match_rate, loocv_results_df = perform_loocv_for_k_detailed(
                scores_series, k1, k2, n_min=n_min, output_dir=output_dir
            )

            if np.isnan(mean_change_rate) or np.isnan(left_out_match_rate):
                continue

            search_results.append({
                'k1': float(k1), 'k2': float(k2),
                'Silhouette_Curve': float(sil_curve),
                'Davies_Bouldin_Curve': float(db_curve),
                'Cohen_d_Curve': float(effect_size_curve),
                'Balance_Score': float(balance),
                'LOOCV_Mean_Change_Rate': float(mean_change_rate),
                'LOOCV_Mean_Match_Rate': float(left_out_match_rate)
            })
            valid_count += 1

    if verbose:
        print(f"\nFinished grid search. Valid combos: {valid_count}")
    if valid_count < 10:
        print("Warning: Low valid k1/k2 combinations. Consider adjusting parameters.")
    if not search_results:
        print("No valid (k1,k2) found. Using default k1=1.0, k2=1.0")
        return 1.0, 1.0, pd.DataFrame()

    results_df_log = pd.DataFrame(search_results)

    db_normalized = 1 / (1 + results_df_log['Davies_Bouldin_Curve'])
    sep_metrics_curve = results_df_log[['Silhouette_Curve', 'Cohen_d_Curve']].copy()
    sep_metrics_curve['Davies_Bouldin_Curve'] = db_normalized
    sep_scaled_curve = StandardScaler().fit_transform(sep_metrics_curve.values)
    results_df_log['CSS'] = (1/3) * sep_scaled_curve[:, 0] + (1/3) * sep_scaled_curve[:, 1] + (1/3) * sep_scaled_curve[:, 2]

    results_df_log['CBS'] = StandardScaler().fit_transform(results_df_log[['Balance_Score']].values).flatten()

    results_df_log['Effective_Change_Rate'] = 1.0 - results_df_log['LOOCV_Mean_Change_Rate']
    rob_metrics = results_df_log[['Effective_Change_Rate', 'LOOCV_Mean_Match_Rate']].copy()
    rob_scaled = StandardScaler().fit_transform(rob_metrics.values)
    results_df_log['CRS'] = 0.5 * rob_scaled[:, 0] + 0.5 * rob_scaled[:, 1]

    results_df_log['ICES'] = w_separation * results_df_log['CSS'] + \
                             w_balance * results_df_log['CBS'] + \
                             w_robustness * results_df_log['CRS']

    best_idx = results_df_log['ICES'].idxmax()
    top_val = results_df_log.loc[best_idx, 'ICES']
    top_df = results_df_log[results_df_log['ICES'] == top_val].copy()
    top_df['k_sum'] = top_df['k1'] + top_df['k2']
    best_row = top_df.loc[top_df['k_sum'].idxmin()]
    best_k1 = float(best_row['k1'])
    best_k2 = float(best_row['k2'])

    results_sorted = results_df_log.sort_values('ICES', ascending=False).reset_index(drop=True)
    results_sorted.to_csv("search_k_result.csv", index=False)
    print("Saved ICES search results to search_k_result.csv")

    return best_k1, best_k2, results_sorted

# 执行网格搜索
scores_series = remaining_df['CRS']
output_dir = "k_LOOCV_Results"
best_k1, best_k2, log_df = find_optimal_k_with_css_cbs_crs(
    scores_series,
    curve_features_df,
    metrics_df,
    w_separation=1/3,
    w_balance=1/3,
    w_robustness=1/3,
    n_min=1,
    k_min=0.05,
    k_max=1.5,
    k_step=0.05,
    verbose=True,
    output_dir=output_dir
)

best_ices = log_df[
    (log_df['k1'] == best_k1) & (log_df['k2'] == best_k2)
]['ICES'].values[0]

print(f"\n✅ Best k1={best_k1:.2f}, k2={best_k2:.2f}, ICES={best_ices:.4f}")






# ======================
# Step 6. CRS-ASI Classification and LOOCV Analysis
# ======================

# --- Output directory ---
output_dir = "CRS_ASI_LOOCV_Results"
Path(output_dir).mkdir(exist_ok=True)

# --- Load CRS data ---
df = pd.read_csv("use_metrics.csv", index_col=0)
crs_series = df['CRS']

# --- Best k1/k2 from Step5 ---
k1 = best_k1  # <- 从 Step5 搜索结果传入
k2 = best_k2

# --- CRS-ASI classification function ---
def crs_asi_classification(scores_series, k1, k2, apply_correction=True):
    scores = np.asarray(scores_series)
    mean = np.mean(scores)
    std = np.std(scores)
    if apply_correction and 1.5 < len(scores) < 30:
        std = std * np.sqrt(len(scores) / (len(scores) - 1.5))
    low_thresh = mean - k1 * std
    high_thresh = mean + k2 * std
    groups = np.select(
        [scores <= low_thresh, scores >= high_thresh],
        ['Low', 'High'],
        default='Intermediate'
    )
    return groups, {'mean': mean, 'std': std, 'low_thresh': low_thresh, 'high_thresh': high_thresh}

# --- Classification ---
crs_groups, threshold_info = crs_asi_classification(crs_series, k1, k2)
low_thresh, high_thresh = threshold_info['low_thresh'], threshold_info['high_thresh']
df['CRS_ASI'] = crs_groups

# --- LOOCV ---
def perform_loocv_crs_asi(scores_series, k1, k2, n_min=1, output_dir=output_dir):
    patient_ids = scores_series.index.to_list()
    loocv_results = []
    consistency_records = []
    left_out_match_flags = []

    original_groups_series = pd.Series(crs_groups, index=patient_ids)

    for left_out_id in patient_ids:
        loo_scores = scores_series.drop(labels=left_out_id)
        if len(loo_scores) < 3:
            continue

        # Recompute thresholds without left-out patient
        loo_groups, loo_thresh = crs_asi_classification(loo_scores, k1, k2)
        loo_low, loo_high = loo_thresh['low_thresh'], loo_thresh['high_thresh']
        loo_groups_series = pd.Series(loo_groups, index=loo_scores.index)

        # Skip if any group < n_min or less than 3 groups
        unique_loo, counts_loo = np.unique(loo_groups, return_counts=True)
        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            continue

        # 留出患者分类
        left_out_score = float(scores_series.loc[left_out_id])
        if left_out_score <= loo_low:
            loo_label_leftout = 'Low'
        elif left_out_score >= loo_high:
            loo_label_leftout = 'High'
        else:
            loo_label_leftout = 'Intermediate'

        orig_label_leftout = original_groups_series.loc[left_out_id]
        left_out_match = int(loo_label_leftout == orig_label_leftout)
        left_out_match_flags.append(left_out_match)

        # 记录剩余患者结果
        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_scores.index,
            'CRS': loo_scores.values,
            'LOOCV_Group': loo_groups_series.values,
            'Original_Group': original_groups_series.loc[loo_scores.index].values,
            'Low_Threshold': loo_low,
            'High_Threshold': loo_high,
            'Left_Out_ID': left_out_id,
            'Left_Out_Score': left_out_score,
            'Left_Out_LOOCV_Group': loo_label_leftout,
            'Left_Out_Original_Group': orig_label_leftout,
            'Left_Out_Match': left_out_match
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_filename = f"{output_dir}/LOOCV_Result_Patient_{left_out_id}.csv"
        loo_result_df.to_csv(loo_filename, index=False)
        loocv_results.append(loo_result_df)

        # 添加一致性记录
        for pid in loo_scores.index:
            changed_flag = int(original_groups_series.loc[pid] != loo_groups_series.loc[pid])
            consistency_records.append(changed_flag)

    mean_change_rate = float(np.mean(consistency_records)) if consistency_records else np.nan
    mean_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()
    return mean_change_rate, mean_match_rate, loocv_results_df

mean_change_rate, mean_match_rate, loocv_results_df = perform_loocv_crs_asi(crs_series, k1, k2, output_dir=output_dir)

# --- Compute metrics ---
# Silhouette & DBI
labels_codes = df['CRS_ASI'].astype('category').cat.codes
sil_curve = silhouette_score(curve_features_df.values, labels_codes)
db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

# Cohen's d
param_features = df[['MED','IC50','IC90']].copy()
param_features = np.log1p(param_features)
cohens_ds = []
for param in ['MED','IC50','IC90']:
    param_values = param_features[param].values
    group_values = [param_values[labels_codes==i] for i in range(3)]
    for g1, g2 in combinations(range(3), 2):
        std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
        n1, n2 = len(group_values[g1]), len(group_values[g2])
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if n1+n2>2 else 0
        cohens_d = (np.mean(group_values[g1])-np.mean(group_values[g2])) / pooled_std if pooled_std>0 else 0.0
        cohens_ds.append(abs(cohens_d))
effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

# Balance score
def calculate_balance_score(groups):
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs>0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy>0 else 0.0
balance = calculate_balance_score(df['CRS_ASI'])
Effective_Change_Rate = 1.0 - mean_change_rate

# --- Save metrics ---
metrics_df = pd.DataFrame([{
    'Algorithm': 'CRS_ASI',
    'Silhouette_Curve': sil_curve,
    'Davies_Bouldin_Curve': db_curve,
    'Cohen_d_Curve': effect_size_curve,
    'Balance_Score': balance,
    'LOOCV_Mean_Change_Rate': mean_change_rate,
    'LOOCV_Mean_Match_Rate': mean_match_rate,
    'Effective_Change_Rate': Effective_Change_Rate
}])
metrics_df.to_csv(f"CRS_ASI_Classification_Metrics.csv", index=False)

# --- Save classification and LOOCV results ---
df.to_csv("CRS_ASI_classification_results.csv")
loocv_results_df.to_csv(f"{output_dir}/LOOCV_All_Results.csv", index=False)

# --- Visualization ---
sns.set(style="whitegrid")

# 1. CRS distribution
plt.figure(figsize=(20,6))
plt.suptitle('CDDP CRS-ASI Classification Analysis', fontsize=14, y=1.02)

plt.subplot(1,3,1)
sns.kdeplot(crs_series, fill=True, color='#4e79a7', alpha=0.5)
plt.axvline(low_thresh, color='g', linestyle='--', label=f'Low-sensitivity: {low_thresh:.2f}')
plt.axvline(high_thresh, color='r', linestyle='--', label=f'High-sensitivity: {high_thresh:.2f}')
plt.title('CRS Distribution', fontsize=12)
plt.xlabel('CRS')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.2)

# 2. Patient grouping scatter plot
plt.subplot(1,3,2)
crs_output_df = df[['CRS','CRS_ASI']].copy()
group_colors = {'Low':'#e15759', 'Intermediate':'#f28e2b', 'High':'#4e79a7'}
for group, color in group_colors.items():
    mask = crs_output_df['CRS_ASI']==group
    plt.scatter(crs_output_df.index[mask], crs_output_df.loc[mask,'CRS'],
                label=group, c=color, alpha=0.7, s=50)
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.5)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.5)
plt.title('Patient Grouping by CRS-ASI', fontsize=12)
plt.xlabel('Patient ID')
plt.ylabel('CRS')
plt.legend(loc='upper left')
plt.grid(alpha=0.2)

# 3. Group box plot
plt.subplot(1,3,3)
sns.boxplot(x='CRS_ASI', y='CRS', data=crs_output_df,
            order=['Low','Intermediate','High'],
            palette=['#e15759','#f28e2b','#4e79a7'])
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.3)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.3)
plt.title('CRS Distribution by Group', fontsize=12)
plt.xlabel('Sensitivity Group')
plt.ylabel('CRS')
plt.grid(alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(f"CRS_ASI_Classification_Visualization.png", dpi=300, bbox_inches='tight')
plt.show()

# --- LOOCV visualization ---
if not loocv_results_df.empty:
    plt.figure(figsize=(20,12))
    plt.suptitle('CDDP CRS-ASI LOOCV Analysis', fontsize=16, y=1.02)

    # Threshold variation
    threshold_stats_vis = loocv_results_df.groupby('Left_Out_ID').agg({
        'Low_Threshold':'mean',
        'High_Threshold':'mean'
    }).reset_index()
    plt.subplot(2,2,1)
    plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['Low_Threshold'], label='Mean Low Threshold', color='#4e79a7')
    plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['High_Threshold'], label='Mean High Threshold', color='#e15759')
    plt.title('Threshold Variation Across LOOCV', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Threshold Value')
    plt.legend()
    plt.grid(alpha=0.2)

    # Classification change heatmap
    pivot_table_crs = loocv_results_df.pivot_table(index='Patient_ID', columns='Left_Out_ID', values='Changed', aggfunc='mean')
    plt.subplot(2,2,2)
    sns.heatmap(pivot_table_crs, cmap='RdYlGn_r', cbar_kws={'label':'Change Magnitude (0=No,1=Yes)'}, annot=True, fmt='.2f')
    plt.title('Classification Change Heatmap', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Patient ID')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # LOOCV group distribution
    plt.subplot(2,2,3)
    sns.countplot(data=loocv_results_df, x='LOOCV_Group', hue='Original_Group',
                  palette=['#e15759','#f28e2b','#4e79a7'])
    plt.title('LOOCV Group vs Original Group Distribution', fontsize=12)
    plt.xlabel('LOOCV Group')
    plt.ylabel('Count')
    plt.legend(title='Original Group')
    plt.grid(alpha=0.2)

    # Left-out patient match rate
    left_out_summary_vis = loocv_results_df[['Left_Out_ID','Left_Out_Match']].groupby('Left_Out_ID')['Left_Out_Match'].mean().reset_index()
    plt.subplot(2,2,4)
    sns.barplot(data=left_out_summary_vis, x='Left_Out_ID', y='Left_Out_Match', color='#4e79a7')
    plt.title('Left-Out Patient Match Rate Across LOOCV', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Match Rate')
    plt.grid(alpha=0.2)
    plt.xticks(rotation=45)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/CRS_ASI_LOOCV_Analysis_Visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

print(f"\n✅ CRS-ASI LOOCV results and visualizations saved in {output_dir}/")


# Classification Visualization-model curve fitting

# Load data
fitted_responses = pd.read_csv("fitted_data.csv")  # Contains Patient_ID and dose-response columns
classification_results = pd.read_csv("CRS_ASI_classification_results.csv")  # Contains Patient_ID, CRS_ASI, etc.
#preclassified_patients = pd.read_csv("5-FU_preclassified_patients.csv")  # Contains Patient_ID, MSSCF

# Step 1: Extract relevant columns from preclassified patients
#preclassified_df = preclassified_patients[['Patient_ID', 'Pre_C']].copy()

# Step 2: Merge classification_results with preclassified_patients to create a complete classification column
# Use 'CRS_ASI' as the final column name, prioritizing preclassified_patients' MSSCF
#classification_combined = classification_results[['Patient_ID', 'CRS_ASI']].copy()
#classification_combined = classification_combined.merge(
    #preclassified_df,
    #on='Patient_ID',
    #how='outer'  # Ensure all patients from both sources are included
#)

# Create a unified classification column (CRS_ASI)
# If a patient is in preclassified_patients, use Pre_C; otherwise, use CRS_ASI
#classification_combined['CRS_ASI'] = classification_combined['Pre_C'].fillna(classification_combined['CRS_ASI'])
#classification_combined = classification_combined[['Patient_ID', 'CRS_ASI']].dropna(subset=['CRS_ASI'])  # Drop rows with no classification

# Step 3: Merge with fitted_responses to include all dose-response data
#all_patients_df = pd.merge(
    #fitted_responses,
    #classification_combined,
    #on="Patient_ID",
    #how="left"  # Keep all patients from fitted_responses
#)

# Merge classification results with fitted responses
all_patients_df = pd.merge(
    fitted_responses,
    classification_results[['Patient_ID', 'CRS_ASI']],
    on="Patient_ID",
    how="left"
)

# Step 4: Save merged data
all_patients_df.to_csv("classification and data.csv", index=False)
print("✅ Integrated classification (including preclassified patients) and fitted response data saved to '5-FU MSSCF classification and data.csv'")

def plot_msscf_grouped_fitted_curves(sigmoid_results, all_patients_df, dose_range=(0, 50), num_points=200):
    """
    Plot fitted Sigmoid Emax curves for all patients, color-coded by MSSCF group,
    include mean curves for each MSSCF group, and add original data points.

    Parameters:
    - sigmoid_results: Dictionary containing fitted Sigmoid Emax parameters for each patient
    - all_patients_df: DataFrame containing Patient_ID, MSSCF, and fitted responses
    - dose_range: Tuple of (min_dose, max_dose) for plotting
    - num_points: Number of points to generate for smooth curves
    """

    # Define Sigmoid Emax function
    def sigmoid_emax(x, ec50, e0, emax, h):
        return e0 + emax * (x ** h) / (ec50 ** h + x ** h)

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Define bright colors for MSSCF groups
    group_colors = {
        'Low': '#1f77b4',  # Bright blue
        'Intermediate': '#2ca02c',  # Bright green
        'High': '#ff4040'  # Bright red
    }

    # Create unified dose range for plotting
    x_fit = np.linspace(dose_range[0], dose_range[1], num_points)

    # Initialize lists to store responses for mean curve calculation
    group_responses = {'Low': [], 'Intermediate': [], 'High': []}

    # Define dose points available in all_patients_df
    dose_columns = [col for col in all_patients_df.columns if col.startswith('Response_')]
    dose_values = [float(col.replace('Response_', '')) for col in dose_columns]

    # Plot individual patient curves and original data points
    for pid in all_patients_df['Patient_ID']:
        # Get MSSCF group
        msscf_group = all_patients_df[all_patients_df['Patient_ID'] == pid]['CRS_ASI'].iloc[0]
        if pd.isna(msscf_group):
            continue  # Skip patients with missing MSSCF classification

        # Get fitted parameters
        res = sigmoid_results.get(pid)
        if res is None or not res['Fit_Success']:
            continue  # Skip patients with failed fits

        # Extract parameters
        ec50, e0, emax, h = res['params']

        # Calculate fitted curve
        y_fit = sigmoid_emax(x_fit, ec50, e0, emax, h)
        y_fit = np.clip(y_fit, 0, 1)  # Clip to match data preprocessing

        # Store for mean calculation
        group_responses[msscf_group].append(y_fit)

        # Plot individual curve with thicker line
        plt.plot(x_fit, y_fit, color=group_colors[msscf_group], alpha=0.5,
                 linewidth=2, label=None)  # No label for individual curves to avoid clutter

        # Plot original data points
        patient_data = all_patients_df[all_patients_df['Patient_ID'] == pid]
        for dose, col in zip(dose_values, dose_columns):
            if dose <= dose_range[1]:  # Only plot points within dose range
                response = patient_data[col].iloc[0]
                if not pd.isna(response):
                    plt.scatter([dose], [response], color=group_colors[msscf_group],
                                s=50, edgecolor='white', zorder=3, alpha=0.8)

    # Calculate and plot mean curves for each MSSCF group
    for group in group_responses:
        if group_responses[group]:  # Only plot if there are curves in the group
            mean_curve = np.mean(group_responses[group], axis=0)
            plt.plot(x_fit, mean_curve, color=group_colors[group], linewidth=4,
                     label=f'{group} Mean Curve (n={len(group_responses[group])})')

    # Chart decorations
    plt.title('CDDP Fitted Sigmoid Emax Curves by Grouping (0-50 μM)', fontsize=14, pad=20)
    plt.xlabel('Dose (μM)', fontsize=12)
    plt.ylabel('Response', fontsize=12)
    plt.xlim(dose_range[0], dose_range[1])
    plt.ylim(-0.05, 1.2)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axhline(1, color='gray', linestyle=':', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, framealpha=0.8)

    # Save plot
    output_path = "grouped_fitted_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {output_path}")

# Execute the plotting function
if __name__ == "__main__":
    plot_msscf_grouped_fitted_curves(sigmoid_results, all_patients_df, dose_range=(0, 50))






# ======================
# Step 7. IC50-CI Classification and LOOCV Analysis
# ======================

# --- Create output directory ---
output_dir = "IC50-CI_LOOCV_Results"
Path(output_dir).mkdir(exist_ok=True)

# --- Load IC50 data ---
#df = pd.read_csv("5-FU_after_preclassification.csv", index_col=0)  # Patient_ID as index
df = pd.read_csv("use_metrics.csv", index_col=0)  # Patient_ID as index
ic50_series = df['IC50']

# --- Bootstrap to calculate 95% CI ---
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    boot_samples = np.random.choice(data, size=(n_boot, len(data)), replace=True)
    means = np.mean(boot_samples, axis=1)
    lower = np.percentile(means, (1-ci)/2*100)
    upper = np.percentile(means, (1+ci)/2*100)
    return lower, upper

low_thresh, high_thresh = bootstrap_ci(ic50_series.values, n_boot=5000, ci=0.95)
print(f"Bootstrap 95% CI thresholds: Low={low_thresh:.4f}, High={high_thresh:.4f}")

# --- Classification based on CI ---
def ic50_ci_group(ic50):
    if ic50 > high_thresh:
        return 'Low'  # low-sensitivity
    elif ic50 < low_thresh:
        return 'High'  # high-sensitivity
    else:
        return 'Intermediate'

ic50_groups = ic50_series.apply(ic50_ci_group)
df['IC50_CI'] = ic50_groups

# --- LOOCV ---
def perform_loocv_ic50(scores_series, n_min=1, output_dir=output_dir):
    patient_ids = scores_series.index.to_list()
    loocv_results = []
    left_out_match_flags = []

    for left_out_id in patient_ids:
        loo_scores = scores_series.drop(labels=left_out_id)
        # Recompute CI thresholds without left-out patient
        loo_low, loo_high = bootstrap_ci(loo_scores.values, n_boot=5000, ci=0.95)
        loo_groups = loo_scores.apply(lambda x: 'Low' if x > loo_high else 'High' if x < loo_low else 'Intermediate')
        original_groups_series = df.loc[loo_scores.index, 'IC50_CI']

        # Skip if any group has < n_min patients or less than 3 groups
        counts = loo_groups.value_counts()
        if counts.min() < n_min or len(counts) < 3:
            continue

        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_scores.index,
            'IC50': loo_scores.values,
            'LOOCV_Group': loo_groups.values,
            'Original_Group': original_groups_series.values
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_result_df['Left_Out_ID'] = left_out_id
        loocv_results.append(loo_result_df)

        left_out_score = scores_series.loc[left_out_id]
        left_out_label = 'Low' if left_out_score > loo_high else 'High' if left_out_score < loo_low else 'Intermediate'
        left_out_match_flags.append(df.loc[left_out_id,'IC50_CI'] == left_out_label)

    mean_change_rate = float(np.mean([row['Changed'] for df_loo in loocv_results for _, row in df_loo.iterrows()])) if loocv_results else np.nan
    mean_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()
    return mean_change_rate, mean_match_rate, loocv_results_df

mean_change_rate, mean_match_rate, loocv_results_df = perform_loocv_ic50(ic50_series, output_dir=output_dir)

# --- Compute basic metrics ---
labels_codes = ic50_groups.astype('category').cat.codes
sil_curve = silhouette_score(curve_features_df.values, labels_codes)
db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

# Cohen's d
param_features = df.loc[ic50_series.index, ['MED','IC50','IC90']].copy()
param_features = np.log1p(param_features)
cohens_ds = []
for param in ['MED','IC50','IC90']:
    param_values = param_features[param].values
    group_values = [param_values[labels_codes == i] for i in range(3)]
    for g1, g2 in combinations(range(3), 2):
        std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
        n1, n2 = len(group_values[g1]), len(group_values[g2])
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if n1+n2>2 else 0
        cohens_d = (np.mean(group_values[g1]) - np.mean(group_values[g2])) / pooled_std if pooled_std>0 else 0.0
        cohens_ds.append(abs(cohens_d))
effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

# Balance
def calculate_balance_score(groups):
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs>0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy>0 else 0.0

balance = calculate_balance_score(ic50_groups)
Effective_Change_Rate = 1.0 - mean_change_rate

# --- Save metrics ---
metrics_df = pd.DataFrame([{
    'Algorithm': 'IC50_CI',
    'Silhouette_Curve': sil_curve,
    'Davies_Bouldin_Curve': db_curve,
    'Cohen_d_Curve': effect_size_curve,
    'Balance_Score': balance,
    'LOOCV_Mean_Change_Rate': mean_change_rate,
    'LOOCV_Mean_Match_Rate': mean_match_rate,
    'Effective_Change_Rate': Effective_Change_Rate
}])
metrics_df.to_csv(f"IC50_CI_Classification_Metrics.csv", index=False)

# --- Save classification and LOOCV results ---
df.to_csv("IC50_CI_classification_results.csv")
loocv_results_df.to_csv(f"{output_dir}/LOOCV_All_Results.csv", index=False)

print("\n✅ IC50-CI LOOCV results saved.")

sns.set(style="whitegrid")

# --- IC50-CI Classification Visualization ---
plt.figure(figsize=(20, 6))
plt.suptitle('CDDP IC50-CI Classification Analysis', fontsize=14, y=1.02)

# 1. IC50 distribution
plt.subplot(1, 3, 1)
sns.kdeplot(ic50_series, fill=True, color='#4e79a7', alpha=0.5)
plt.axvline(low_thresh, color='g', linestyle='--', label=f'High-sensitivity: {low_thresh:.2f}')
plt.axvline(high_thresh, color='r', linestyle='--', label=f'Low-sensitivity: {high_thresh:.2f}')
plt.title('IC50 Distribution', fontsize=12)
plt.xlabel('IC50 (μM)')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.2)

# 2. Patient grouping scatter plot
plt.subplot(1, 3, 2)
ic50_output_df = df[['IC50','IC50_CI']].copy()
group_colors = {'Low':'#e15759', 'Intermediate':'#f28e2b', 'High':'#4e79a7'}
for group, color in group_colors.items():
    mask = ic50_output_df['IC50_CI'] == group
    plt.scatter(ic50_output_df.index[mask], ic50_output_df.loc[mask,'IC50'],
                label=group, c=color, alpha=0.7, s=50)
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.5)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.5)
plt.title('Patient Grouping by IC50-CI', fontsize=12)
plt.xlabel('Patient ID')
plt.ylabel('IC50 (μM)')
plt.legend(loc='upper left')
plt.grid(alpha=0.2)

# 3. Group box plot
plt.subplot(1, 3, 3)
sns.boxplot(x='IC50_CI', y='IC50', data=ic50_output_df,
            order=['High','Intermediate','Low'],
            palette=['#4e79a7','#f28e2b','#e15759'])
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.3)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.3)
plt.title('IC50 Distribution by Group', fontsize=12)
plt.xlabel('Sensitivity Group')
plt.ylabel('IC50 (μM)')
plt.grid(alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(f"IC50_CI_Classification_Visualization.png", dpi=300, bbox_inches='tight')
plt.show()


# --- LOOCV Visualization ---
plt.figure(figsize=(20, 12))
plt.suptitle('CDDP IC50-CI LOOCV Analysis', fontsize=16, y=1.02)

# 1. Classification change heatmap
pivot_table_ic50 = loocv_results_df.pivot_table(index='Patient_ID', columns='Left_Out_ID', values='Changed', aggfunc='mean')
plt.subplot(2, 2, 1)
sns.heatmap(pivot_table_ic50, cmap='RdYlGn_r', cbar_kws={'label': 'Changed (1=Yes, 0=No)'}, annot=True, fmt='.0f')
plt.title('Classification Change Heatmap', fontsize=12)
plt.xlabel('Left-Out Patient ID')
plt.ylabel('Patient ID')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. LOOCV group distribution
plt.subplot(2, 2, 2)
sns.countplot(data=loocv_results_df, x='LOOCV_Group', hue='Original_Group',
              palette=['#4e79a7','#f28e2b','#e15759'], order=['High','Intermediate','Low'])
plt.title('LOOCV Group vs Original Group Distribution', fontsize=12)
plt.xlabel('LOOCV Group')
plt.ylabel('Count')
plt.legend(title='Original Group')
plt.grid(alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(f"{output_dir}/IC50_LOOCV_Analysis_Visualization.png", dpi=300, bbox_inches='tight')
plt.show()






# ======================
# Step 8. CRS-CI Classification and LOOCV Analysis
# ======================

# --- Create output directory ---
output_dir = "CRS_CI_LOOCV_Results"
Path(output_dir).mkdir(exist_ok=True)

# --- Load remaining patients data ---
df = pd.read_csv("use_metrics.csv", index_col=0)  # Patient_ID as index
crs_series = df['CRS']

# --- Bootstrap to calculate 95% CI ---
def bootstrap_ci(data, n_boot=5000, ci=0.95):
    boot_samples = np.random.choice(data, size=(n_boot, len(data)), replace=True)
    means = np.mean(boot_samples, axis=1)
    lower = np.percentile(means, (1-ci)/2*100)
    upper = np.percentile(means, (1+ci)/2*100)
    return lower, upper

low_thresh, high_thresh = bootstrap_ci(crs_series.values, n_boot=5000, ci=0.95)
print(f"Bootstrap 95% CI thresholds for CRS: Low={low_thresh:.4f}, High={high_thresh:.4f}")

# --- Classification based on CRS CI ---
def crs_ci_group(crs):
    if crs > high_thresh:
        return 'High'
    elif crs < low_thresh:
        return 'Low'
    else:
        return 'Intermediate'

crs_groups = crs_series.apply(crs_ci_group)
df['CRS_CI'] = crs_groups

# --- LOOCV ---
def perform_loocv_crs(scores_series, n_min=2, output_dir=output_dir):
    patient_ids = scores_series.index.to_list()
    loocv_results = []
    consistency_records = []
    left_out_match_flags = []

    original_groups = scores_series.apply(crs_ci_group)
    original_groups_series = pd.Series(original_groups, index=patient_ids)

    for left_out_id in patient_ids:
        loo_scores = scores_series.drop(labels=left_out_id)
        if len(loo_scores) < 3:
            continue

        # Recompute CI thresholds without left-out patient
        loo_low, loo_high = bootstrap_ci(loo_scores.values, n_boot=5000, ci=0.95)

        def loo_crs_ci_group(crs):
            if crs > loo_high:
                return 'High'
            elif crs < loo_low:
                return 'Low'
            else:
                return 'Intermediate'

        loo_groups = loo_scores.apply(loo_crs_ci_group)
        loo_groups_series = pd.Series(loo_groups, index=loo_scores.index)

        # Skip if any group has < n_min patients or less than 3 groups
        unique_loo, counts_loo = np.unique(loo_groups, return_counts=True)
        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            continue

        # 计算留出患者的分类
        left_out_score = float(scores_series.loc[left_out_id])
        loo_label_leftout = loo_crs_ci_group(left_out_score)
        orig_label_leftout = original_groups_series.loc[left_out_id]
        left_out_match = int(loo_label_leftout == orig_label_leftout)
        left_out_match_flags.append(left_out_match)

        # 记录剩余患者结果，并添加留出患者信息
        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_scores.index,
            'CRS': loo_scores.values,
            'LOOCV_Group': loo_groups_series.values,
            'Original_Group': original_groups_series.loc[loo_scores.index].values,
            'Low_Threshold': loo_low,
            'High_Threshold': loo_high,
            'Left_Out_ID': left_out_id,
            'Left_Out_Score': left_out_score,
            'Left_Out_LOOCV_Group': loo_label_leftout,
            'Left_Out_Original_Group': orig_label_leftout,
            'Left_Out_Match': left_out_match
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_filename = f"{output_dir}/LOOCV_Result_Patient_{left_out_id}.csv"
        loo_result_df.to_csv(loo_filename, index=False)
        loocv_results.append(loo_result_df)

        # 添加一致性记录（剩余患者）
        for pid in loo_scores.index:
            changed_flag = int(original_groups_series.loc[pid] != loo_groups_series.loc[pid])
            consistency_records.append(changed_flag)

    mean_change_rate = float(np.mean(consistency_records)) if consistency_records else np.nan
    mean_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()
    return mean_change_rate, mean_match_rate, loocv_results_df

mean_change_rate, mean_match_rate, loocv_results_df = perform_loocv_crs(crs_series, output_dir=output_dir)

# --- Compute basic metrics ---
labels_codes = crs_groups.astype('category').cat.codes
sil_curve = silhouette_score(curve_features_df.values, labels_codes)
db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

# Cohen's d
param_features = df.loc[crs_series.index, ['MED','IC50','IC90']].copy()
param_features = np.log1p(param_features)
cohens_ds = []
for param in ['MED','IC50','IC90']:
    param_values = param_features[param].values
    group_values = [param_values[labels_codes == i] for i in range(3)]
    for g1, g2 in combinations(range(3), 2):
        std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
        n1, n2 = len(group_values[g1]), len(group_values[g2])
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if n1+n2>2 else 0
        cohens_d = (np.mean(group_values[g1]) - np.mean(group_values[g2])) / pooled_std if pooled_std>0 else 0.0
        cohens_ds.append(abs(cohens_d))
effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

# Balance
def calculate_balance_score(groups):
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs>0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy>0 else 0.0

balance = calculate_balance_score(crs_groups)
Effective_Change_Rate = 1.0 - mean_change_rate

# --- Save metrics ---
metrics_df = pd.DataFrame([{
    'Algorithm': 'CRS_CI',
    'Silhouette_Curve': sil_curve,
    'Davies_Bouldin_Curve': db_curve,
    'Cohen_d_Curve': effect_size_curve,
    'Balance_Score': balance,
    'LOOCV_Mean_Change_Rate': mean_change_rate,
    'LOOCV_Mean_Match_Rate': mean_match_rate,
    'Effective_Change_Rate': Effective_Change_Rate
}])
metrics_df.to_csv(f"CRS_CI_Classification_Metrics.csv", index=False)

# --- Save classification and LOOCV results ---
df.to_csv("CRS_CI_classification_results.csv")
loocv_results_df.to_csv(f"{output_dir}/LOOCV_All_Results.csv", index=False)

# Save LOOCV summary
summary_df = pd.DataFrame({
    'Metric': ['Mean_Change_Rate', 'Mean_Match_Rate'],
    'Value': [mean_change_rate, mean_match_rate]
})
summary_df.to_csv(f"{output_dir}/LOOCV_Summary.csv", index=False)

# Calculate threshold statistics
if not loocv_results_df.empty:
    threshold_stats = loocv_results_df.groupby('Left_Out_ID').agg({
        'Low_Threshold': ['mean', 'std', 'min', 'max'],
        'High_Threshold': ['mean', 'std', 'min', 'max'],
        'Left_Out_Match': ['mean']
    }).reset_index()
    threshold_stats.columns = ['Left_Out_ID', 'Low_Mean', 'Low_Std', 'Low_Min', 'Low_Max',
                               'High_Mean', 'High_Std', 'High_Min', 'High_Max',
                               'Left_Out_Match_Rate']
    threshold_stats.to_csv(f"{output_dir}/Threshold_Statistics.csv", index=False)

# Save left-out patient classification summary
left_out_summary = loocv_results_df[['Left_Out_ID', 'Left_Out_Score', 'Left_Out_LOOCV_Group',
                                    'Left_Out_Original_Group', 'Left_Out_Match']].drop_duplicates()
left_out_summary.to_csv(f"{output_dir}/Left_Out_Classification_Summary.csv", index=False)

print("\n✅ CRS-CI classification and LOOCV results saved.")

# --- Visualization ---
sns.set(style="whitegrid")

# 1. CRS distribution
plt.figure(figsize=(20,6))
plt.suptitle('CDDP CRS-CI Classification Analysis', fontsize=14, y=1.02)

plt.subplot(1,3,1)
sns.kdeplot(crs_series, fill=True, color='#4e79a7', alpha=0.5)
plt.axvline(low_thresh, color='g', linestyle='--', label=f'Low-sensitivity: {low_thresh:.2f}')
plt.axvline(high_thresh, color='r', linestyle='--', label=f'High-sensitivity: {high_thresh:.2f}')
plt.title('CRS Distribution', fontsize=12)
plt.xlabel('CRS')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.2)

# 2. Patient grouping scatter plot
plt.subplot(1,3,2)
crs_output_df = df[['CRS','CRS_CI']].copy()
group_colors = {'Low':'#e15759', 'Intermediate':'#f28e2b', 'High':'#4e79a7'}
for group, color in group_colors.items():
    mask = crs_output_df['CRS_CI'] == group
    plt.scatter(crs_output_df.index[mask], crs_output_df.loc[mask,'CRS'],
                label=group, c=color, alpha=0.7, s=50)
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.5)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.5)
plt.title('Patient Grouping by CRS-CI', fontsize=12)
plt.xlabel('Patient ID')
plt.ylabel('CRS')
plt.legend(loc='upper left')
plt.grid(alpha=0.2)

# 3. Group box plot
plt.subplot(1,3,3)
sns.boxplot(x='CRS_CI', y='CRS', data=crs_output_df,
            order=['Low','Intermediate','High'],
            palette=['#e15759','#f28e2b','#4e79a7'])
plt.axhline(low_thresh, color='gray', linestyle='--', alpha=0.3)
plt.axhline(high_thresh, color='gray', linestyle=':', alpha=0.3)
plt.title('CRS Distribution by Group', fontsize=12)
plt.xlabel('Sensitivity Group')
plt.ylabel('CRS')
plt.grid(alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(f"CRS_CI_Classification_Visualization.png", dpi=300, bbox_inches='tight')
plt.show()

# --- LOOCV Visualization ---
if not loocv_results_df.empty:
    plt.figure(figsize=(20, 12))
    plt.suptitle('CDDP CRS-CI LOOCV Analysis', fontsize=16, y=1.02)

    # 1. Threshold variation plot
    threshold_stats_vis = loocv_results_df.groupby('Left_Out_ID').agg({
        'Low_Threshold': 'mean',
        'High_Threshold': 'mean'
    }).reset_index()
    plt.subplot(2, 2, 1)
    plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['Low_Threshold'], label='Mean Low Threshold', color='#4e79a7')
    plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['High_Threshold'], label='Mean High Threshold', color='#e15759')
    plt.title('Threshold Variation Across LOOCV', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Threshold Value')
    plt.legend()
    plt.grid(alpha=0.2)

    # 2. Classification change heatmap
    pivot_table_crs = loocv_results_df.pivot_table(index='Patient_ID', columns='Left_Out_ID', values='Changed', aggfunc='mean')
    plt.subplot(2, 2, 2)
    sns.heatmap(pivot_table_crs, cmap='RdYlGn_r', cbar_kws={'label': 'Change Magnitude (0=No, 1=Yes)'}, annot=True, fmt='.2f')
    plt.title('Classification Change Heatmap', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Patient ID')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 3. LOOCV group distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=loocv_results_df, x='LOOCV_Group', hue='Original_Group',
                  palette=['#e15759','#f28e2b','#4e79a7'])
    plt.title('LOOCV Group vs Original Group Distribution', fontsize=12)
    plt.xlabel('LOOCV Group')
    plt.ylabel('Count')
    plt.legend(title='Original Group')
    plt.grid(alpha=0.2)

    # 4. Left-out patient match rate
    left_out_summary_vis = loocv_results_df[['Left_Out_ID', 'Left_Out_Match']].groupby('Left_Out_ID')['Left_Out_Match'].mean().reset_index()
    plt.subplot(2, 2, 4)
    sns.barplot(data=left_out_summary_vis, x='Left_Out_ID', y='Left_Out_Match', color='#4e79a7')
    plt.title('Left-Out Patient Match Rate Across LOOCV', fontsize=12)
    plt.xlabel('Left-Out Patient ID')
    plt.ylabel('Match Rate')
    plt.grid(alpha=0.2)
    plt.xticks(rotation=45)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/CRS_LOOCV_Analysis_Visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Boxplot for threshold and parameter statistics
    plt.figure(figsize=(12, 6))
    plt.suptitle('Threshold Statistics Across LOOCV', fontsize=14, y=1.02)

    plt.subplot(1, 2, 1)
    sns.boxplot(data=loocv_results_df[['Low_Threshold', 'High_Threshold']],
                palette=['#4e79a7', '#e15759'])
    plt.title('Low and High Threshold Distribution', fontsize=12)
    plt.ylabel('Threshold Value')
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=loocv_results_df[['CRS']], palette=['#4e79a7'])
    plt.title('CRS Distribution', fontsize=12)
    plt.ylabel('CRS Value')
    plt.grid(alpha=0.2)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{output_dir}/Threshold_Parameter_Boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()

print(f"\nAll LOOCV results and visualizations saved in {output_dir}/")






# ======================
# Step 9. IC50 Clustering with Machine Learning Algorithms and LOOCV Analysis
# ======================

# --- Load IC50 data ---
df = pd.read_csv("use_metrics.csv", index_col=0)  # Patient_ID as index
ic50_series = df['IC50']
ic50_array = ic50_series.values.reshape(-1, 1)  # Reshape for sklearn (1D feature)

try:
    curve_features_df
except NameError:
    curve_features_df = pd.DataFrame({'IC50': ic50_series})

# --- Define clustering algorithms and their output directories ---
algorithms = [
    ('IC50-KMeans', KMeans(n_clusters=3, random_state=42)),
    ('IC50-Hierarchical', AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ('IC50-GMM', GaussianMixture(n_components=3, random_state=42))
]
output_dirs = ['IC50_KMeans_LOOCV_Results', 'IC50_Hierarchical_LOOCV_Results', 'IC50_GMM_LOOCV_Results']

# --- Helper functions ---
def map_cluster_labels_to_groups(labels, ic50_values):
    """Map cluster labels to 'Low', 'Intermediate', 'High' based on IC50 means"""
    unique_labels = np.unique(labels)
    if len(unique_labels) != 3:
        # Fallback: assign based on IC50 quantiles
        quantiles = np.percentile(ic50_values, [33, 66])
        groups = np.where(ic50_values < quantiles[0], 'High',
                         np.where(ic50_values > quantiles[1], 'Low', 'Intermediate'))
        return groups
    cluster_means = {label: np.mean(ic50_values[labels == label]) for label in unique_labels}
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])  # Sort by mean IC50
    label_mapping = {sorted_clusters[0][0]: 'High', sorted_clusters[1][0]: 'Intermediate', sorted_clusters[2][0]: 'Low'}
    return np.array([label_mapping.get(label, 'Intermediate') for label in labels])

def calculate_balance_score(groups):
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0

def compute_cohens_d(param_values, labels_codes):
    cohens_ds = []
    for g1, g2 in combinations(range(len(np.unique(labels_codes))), 2):
        group_values_g1 = param_values[labels_codes == g1]
        group_values_g2 = param_values[labels_codes == g2]
        if len(group_values_g1) > 1 and len(group_values_g2) > 1:
            mean1, mean2 = np.mean(group_values_g1), np.mean(group_values_g2)
            std1, std2 = np.std(group_values_g1, ddof=1), np.std(group_values_g2, ddof=1)
            n1, n2 = len(group_values_g1), len(group_values_g2)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = abs((mean1 - mean2) / pooled_std) if pooled_std > 0 else 0.0
            cohens_ds.append(cohens_d)
    return np.mean(cohens_ds) if cohens_ds else 0.0

# --- LOOCV for clustering ---
def perform_loocv_clustering(ic50_array, model, algo_name, n_min=1, output_dir="IC50_ML_Clustering_LOOCV_Results"):
    Path(output_dir).mkdir(exist_ok=True)
    patient_ids = ic50_series.index.to_list()
    loocv_results = []
    consistency_records = []
    left_out_match_flags = []
    skipped_count = 0
    valid_count = 0

    # Original clustering
    model.fit(ic50_array)
    original_labels = model.predict(ic50_array) if algo_name == 'IC50-GMM' else model.labels_
    original_groups = map_cluster_labels_to_groups(original_labels, ic50_series.values)
    original_groups_series = pd.Series(original_groups, index=patient_ids)

    print(f"Starting LOOCV for {algo_name} with {len(patient_ids)} patients...")

    for left_out_id in patient_ids:
        idx = patient_ids.index(left_out_id)
        loo_array = np.delete(ic50_array, idx, axis=0)
        loo_ids = [p for i, p in enumerate(patient_ids) if i != idx]

        if len(loo_array) < 3:
            skipped_count += 1
            print(f"Skipped {left_out_id}: Not enough samples ({len(loo_array)})")
            continue

        # Re-fit model on LOO data
        model.fit(loo_array)
        loo_labels = model.predict(loo_array) if algo_name == 'IC50-GMM' else model.labels_
        loo_groups = map_cluster_labels_to_groups(loo_labels, ic50_series.loc[loo_ids].values)
        loo_groups_series = pd.Series(loo_groups, index=loo_ids)

        # Check groups (use fallback if needed)
        unique_loo, counts_loo = np.unique(loo_groups, return_counts=True)
        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            # Fallback: use quantile-based grouping for LOO
            quantiles = np.percentile(ic50_series.loc[loo_ids].values, [33, 66])
            loo_groups_fallback = np.where(ic50_series.loc[loo_ids].values < quantiles[0], 'High',
                                          np.where(ic50_series.loc[loo_ids].values > quantiles[1], 'Low', 'Intermediate'))
            loo_groups_series = pd.Series(loo_groups_fallback, index=loo_ids)
            unique_loo, counts_loo = np.unique(loo_groups_series, return_counts=True)
            print(f"Fallback used for {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")

        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            skipped_count += 1
            print(f"Skipped {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")
            continue

        valid_count += 1
        print(f"Processed {left_out_id}: groups={unique_loo}, counts={counts_loo}")

        # Assign label to left-out patient
        left_out_ic50 = ic50_array[idx].reshape(1, -1)  # Reshape for GMM predict
        if algo_name == 'IC50-KMeans':
            distances = np.abs(model.cluster_centers_.flatten() - left_out_ic50.flatten()[0])
            loo_label_leftout_idx = np.argmin(distances)
            loo_label_leftout = map_cluster_labels_to_groups(np.array([loo_label_leftout_idx]), np.array([left_out_ic50.flatten()[0]]))[0]
        elif algo_name == 'IC50-GMM':
            loo_label_leftout_idx = model.predict(left_out_ic50)[0]
            loo_label_leftout = map_cluster_labels_to_groups(np.array([loo_label_leftout_idx]), np.array([left_out_ic50.flatten()[0]]))[0]
        else:  # Hierarchical
            # Use nearest neighbor
            distances = np.abs(ic50_series.loc[loo_ids].values - left_out_ic50.flatten()[0])
            nearest_pid = loo_ids[np.argmin(distances)]
            loo_label_leftout = loo_groups_series.loc[nearest_pid]

        orig_label_leftout = original_groups_series.loc[left_out_id]
        left_out_match = int(loo_label_leftout == orig_label_leftout)
        left_out_match_flags.append(left_out_match)

        # Record remaining patients
        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_ids,
            'IC50': ic50_series.loc[loo_ids].values,
            'LOOCV_Group': loo_groups_series.values,
            'Original_Group': original_groups_series.loc[loo_ids].values,
            'Left_Out_ID': left_out_id,
            'Left_Out_Score': left_out_ic50.flatten()[0],
            'Left_Out_LOOCV_Group': loo_label_leftout,
            'Left_Out_Original_Group': orig_label_leftout,
            'Left_Out_Match': left_out_match
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_filename = f"{output_dir}/LOOCV_Result_Patient_{left_out_id}.csv"
        loo_result_df.to_csv(loo_filename, index=False)
        loocv_results.append(loo_result_df)

        # Consistency for remaining patients
        for pid in loo_ids:
            changed_flag = int(original_groups_series.loc[pid] != loo_groups_series.loc[pid])
            consistency_records.append(changed_flag)

    print(f"LOOCV for {algo_name}: Valid iterations = {valid_count}, Skipped = {skipped_count}")
    mean_change_rate = float(np.mean(consistency_records)) if consistency_records else np.nan
    mean_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()
    return mean_change_rate, mean_match_rate, loocv_results_df

# --- Run for each algorithm ---
results_summary = []
for (algo_name, model), out_dir in zip(algorithms, output_dirs):
    print(f"\n--- Running {algo_name} Clustering ---")
    Path(out_dir).mkdir(exist_ok=True)

    # Fit model on full data
    model.fit(ic50_array)
    cluster_labels = model.predict(ic50_array) if algo_name == 'IC50-GMM' else model.labels_
    if len(np.unique(cluster_labels)) != 3:
        print(f"Warning: {algo_name} produced {len(np.unique(cluster_labels))} clusters, using fallback mapping.")
        # Use fallback mapping
        quantiles = np.percentile(ic50_series.values, [33, 66])
        cluster_labels = np.where(ic50_series.values < quantiles[0], 0,
                                 np.where(ic50_series.values > quantiles[1], 2, 1))
    groups = map_cluster_labels_to_groups(cluster_labels, ic50_series.values)
    df[f'{algo_name}_Cluster'] = groups

    # Compute metrics
    labels_codes = pd.Series(groups).astype('category').cat.codes
    sil_curve = silhouette_score(curve_features_df.values, labels_codes)
    db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

    # Cohen's d
    param_features = df.loc[ic50_series.index, ['MED', 'IC50', 'IC90']].copy()
    param_features = np.log1p(param_features)  # Log-transform to stabilize variance
    cohens_ds = []
    for param in ['MED', 'IC50', 'IC90']:
        param_values = param_features[param].values
        group_values = [param_values[labels_codes == i] for i in range(3)]
        for g1, g2 in combinations(range(3), 2):
            std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
            n1, n2 = len(group_values[g1]), len(group_values[g2])
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if n1+n2>2 else 0
            cohens_d = (np.mean(group_values[g1]) - np.mean(group_values[g2])) / pooled_std if pooled_std>0 else 0.0
            cohens_ds.append(abs(cohens_d))
    effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

    # Balance
    def calculate_balance_score(groups):
        unique_labels, counts = np.unique(groups, return_counts=True)
        if len(counts) <= 1:
            return 0.0
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    balance = calculate_balance_score(groups)

    # LOOCV
    mean_change_rate, mean_match_rate, loocv_results_df = perform_loocv_clustering(ic50_array, model, algo_name, n_min=1, output_dir=out_dir)

    effective_change_rate = 1.0 - mean_change_rate if not np.isnan(mean_change_rate) else np.nan

    # Save metrics
    metrics_row = {
        'Algorithm': algo_name,
        'Silhouette_Curve': sil_curve,
        'Davies_Bouldin_Curve': db_curve,
        'Cohen_d_Curve': effect_size_curve,
        'Balance_Score': balance,
        'LOOCV_Mean_Change_Rate': mean_change_rate,
        'LOOCV_Mean_Match_Rate': mean_match_rate,
        'Effective_Change_Rate': effective_change_rate
    }
    results_summary.append(metrics_row)
    pd.DataFrame([metrics_row]).to_csv(f"{algo_name}_Clustering_Metrics.csv", index=False)
    print(f"Saved metrics to {algo_name}_Clustering_Metrics.csv")

    # Save classification
    df.to_csv(f"{algo_name}_Clustering_Results.csv")
    print(f"Saved classification results to {algo_name}_Clustering_Results.csv")

    # Save LOOCV all results
    loocv_results_df.to_csv(f"{out_dir}/LOOCV_All_Results.csv", index=False)
    print(f"Saved LOOCV results to {out_dir}/LOOCV_All_Results.csv")

    # Save LOOCV summary
    summary_df = pd.DataFrame({
        'Metric': ['Mean_Change_Rate', 'Mean_Match_Rate'],
        'Value': [mean_change_rate, mean_match_rate]
    })
    summary_df.to_csv(f"{out_dir}/LOOCV_Summary.csv", index=False)
    print(f"Saved LOOCV summary to {out_dir}/LOOCV_Summary.csv")

    # Cluster centers statistics (if applicable)
    if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
        cluster_centers = model.means_ if algo_name == 'IC50-GMM' else model.cluster_centers_
        cluster_centers_stats = pd.DataFrame({
            'Cluster': ['Cluster_0', 'Cluster_1', 'Cluster_2'],
            'Center_Mean': cluster_centers.flatten()
        })
        cluster_centers_stats.to_csv(f"{out_dir}/IC50_Cluster_Centers_Statistics.csv", index=False)
        print(f"Saved cluster centers to {out_dir}/IC50_Cluster_Centers_Statistics.csv")
    else:
        # Fallback for Hierarchical
        cluster_centers_stats = pd.DataFrame({
            'Cluster': ['Cluster_0', 'Cluster_1', 'Cluster_2'],
            'Center_Mean': [np.mean(ic50_series[groups == g]) for g in ['High', 'Intermediate', 'Low']]
        })
        cluster_centers_stats.to_csv(f"{out_dir}/IC50_Cluster_Centers_Statistics.csv", index=False)
        print(f"Saved cluster centers (fallback) to {out_dir}/IC50_Cluster_Centers_Statistics.csv")

    # Left-out summary
    left_out_summary = loocv_results_df[['Left_Out_ID', 'Left_Out_Score', 'Left_Out_LOOCV_Group',
                                        'Left_Out_Original_Group', 'Left_Out_Match']].drop_duplicates()
    left_out_summary.to_csv(f"{out_dir}/Left_Out_Classification_Summary.csv", index=False)
    print(f"Saved left-out summary to {out_dir}/Left_Out_Classification_Summary.csv")

    print(f"✅ {algo_name} clustering and LOOCV results saved in {out_dir}/")

# Save overall summary
overall_summary = pd.DataFrame(results_summary)
overall_summary.to_csv("IC50_ML_Clustering_Overall_Metrics.csv", index=False)
print("\nOverall clustering metrics saved to IC50_ML_Clustering_Overall_Metrics.csv")

# --- Visualization for each algorithm ---
for (algo_name, model), out_dir in zip(algorithms, output_dirs):
    if f'{algo_name}_Cluster' not in df.columns:
        continue  # Skip if not processed

    sns.set(style="whitegrid")
    groups = df[f'{algo_name}_Cluster']
    ic50_values = ic50_series

    # Classification Visualization
    plt.figure(figsize=(20, 6))
    plt.suptitle(f'{algo_name} Clustering Analysis', fontsize=14, y=1.02)

    # 1. IC50 distribution
    plt.subplot(1, 3, 1)
    sns.kdeplot(ic50_values, fill=True, color='#4e79a7', alpha=0.5)
    # Cluster centers as 'thresholds'
    if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
        centers = model.means_ if algo_name == 'IC50-GMM' else model.cluster_centers_
        centers = centers.flatten()
        plt.axvline(np.min(centers), color='g', linestyle='--', label='Cluster Centers Min')
        plt.axvline(np.max(centers), color='r', linestyle='--', label='Cluster Centers Max')
    plt.title('IC50 Distribution', fontsize=12)
    plt.xlabel('IC50 (μM)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.2)

    # 2. Patient grouping scatter plot
    plt.subplot(1, 3, 2)
    ic50_output_df = df[['IC50', f'{algo_name}_Cluster']].copy()
    ic50_output_df.columns = ['IC50', 'Cluster']
    group_colors = {'Low': '#e15759', 'Intermediate': '#f28e2b', 'High': '#4e79a7'}
    for group, color in group_colors.items():
        mask = ic50_output_df['Cluster'] == group
        plt.scatter(ic50_output_df.index[mask], ic50_output_df.loc[mask, 'IC50'],
                    label=group, c=color, alpha=0.7, s=50)
    plt.title(f'Patient Grouping by {algo_name}', fontsize=12)
    plt.xlabel('Patient ID')
    plt.ylabel('IC50 (μM)')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)

    # 3. Group box plot
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Cluster', y='IC50', data=ic50_output_df,
                order=['High', 'Intermediate', 'Low'],
                palette=['#4e79a7', '#f28e2b', '#e15759'])
    plt.title('IC50 Distribution by Cluster', fontsize=12)
    plt.xlabel('Sensitivity Group')
    plt.ylabel('IC50 (μM)')
    plt.grid(alpha=0.2)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{algo_name}_Clustering_Visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

    # LOOCV Visualization
    loocv_df = pd.read_csv(f"{out_dir}/LOOCV_All_Results.csv")
    if not loocv_df.empty:
        plt.figure(figsize=(20, 12))
        plt.suptitle(f'{algo_name} LOOCV Analysis', fontsize=16, y=1.02)

        # 1. IC50 mean variation plot
        plt.subplot(2, 2, 1)
        threshold_stats_vis = loocv_df.groupby('Left_Out_ID').agg({
            'IC50': 'mean'
        }).reset_index()
        plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['IC50'], label='IC50 Mean', color='#4e79a7')
        plt.title('IC50 Mean Variation Across LOOCV', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('IC50 Value')
        plt.legend()
        plt.grid(alpha=0.2)

        # 2. Classification change heatmap
        pivot_table = loocv_df.pivot_table(index='Patient_ID', columns='Left_Out_ID', values='Changed', aggfunc='mean')
        plt.subplot(2, 2, 2)
        sns.heatmap(pivot_table, cmap='RdYlGn_r', cbar_kws={'label': 'Change Magnitude (0=No, 1=Yes)'}, annot=True, fmt='.2f')
        plt.title('Classification Change Heatmap', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('Patient ID')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 3. LOOCV group distribution
        plt.subplot(2, 2, 3)
        sns.countplot(data=loocv_df, x='LOOCV_Group', hue='Original_Group',
                      palette=['#4e79a7', '#f28e2b', '#e15759'])
        plt.title('LOOCV Group vs Original Group Distribution', fontsize=12)
        plt.xlabel('LOOCV Group')
        plt.ylabel('Count')
        plt.legend(title='Original Group')
        plt.grid(alpha=0.2)

        # 4. Left-out patient match rate
        left_out_summary_vis = loocv_df[['Left_Out_ID', 'Left_Out_Match']].groupby('Left_Out_ID')['Left_Out_Match'].mean().reset_index()
        plt.subplot(2, 2, 4)
        sns.barplot(data=left_out_summary_vis, x='Left_Out_ID', y='Left_Out_Match', color='#4e79a7')
        plt.title('Left-Out Patient Match Rate Across LOOCV', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('Match Rate')
        plt.grid(alpha=0.2)
        plt.xticks(rotation=45)

        plt.tight_layout(pad=2.0)
        plt.savefig(f"{out_dir}/{algo_name}_LOOCV_Analysis_Visualization.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Boxplot for parameters
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{algo_name} Parameter Statistics Across LOOCV', fontsize=14, y=1.02)

        plt.subplot(1, 2, 1)
        sns.boxplot(data=loocv_df[['IC50']], palette=['#4e79a7'])
        plt.title('IC50 Distribution', fontsize=12)
        plt.ylabel('IC50 Value')
        plt.grid(alpha=0.2)

        plt.subplot(1, 2, 2)
        if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
            centers = model.means_ if algo_name == 'IC50-GMM' else model.cluster_centers_
            centers_df = pd.DataFrame({'Cluster_Centers': centers.flatten()})
            sns.boxplot(data=centers_df, palette=['#4e79a7'])
            plt.title('Cluster Centers Distribution', fontsize=12)
            plt.ylabel('Center Value')
        else:
            plt.text(0.5, 0.5, 'No cluster centers available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Cluster Centers (N/A)', fontsize=12)
        plt.grid(alpha=0.2)

        plt.tight_layout(pad=2.0)
        plt.savefig(f"{out_dir}/Parameter_Boxplot.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n===== Step 9 IC50 Clustering with ML Algorithms and LOOCV Completed =====")






# ======================
# Step 10. CRS Clustering with Machine Learning Algorithms and LOOCV Analysis
# ======================

# --- Load CRS data ---
df = pd.read_csv("use_metrics.csv", index_col=0)  # Patient_ID as index
crs_series = df['CRS']
crs_array = crs_series.values.reshape(-1, 1)  # Reshape for sklearn (1D feature)

try:
    curve_features_df
except NameError:
    curve_features_df = pd.DataFrame({'CRS': crs_series})

# --- Define clustering algorithms and their output directories ---
algorithms = [
    ('CRS-KMeans', KMeans(n_clusters=3, random_state=42)),
    ('CRS-Hierarchical', AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ('CRS-GMM', GaussianMixture(n_components=3, random_state=42))
]
output_dirs = ['KMeans_CRS_LOOCV_Results', 'Hierarchical_CRS_LOOCV_Results', 'GMM_CRS_LOOCV_Results']

# --- Helper functions ---
def map_cluster_labels_to_groups(labels, crs_values):
    """Map cluster labels to 'Low', 'Intermediate', 'High' based on CRS means"""
    unique_labels = np.unique(labels)
    if len(unique_labels) != 3:
        # Fallback: assign based on CRS quantiles
        quantiles = np.percentile(crs_values, [33, 66])
        groups = np.where(crs_values < quantiles[0], 'Low',
                         np.where(crs_values > quantiles[1], 'High', 'Intermediate'))
        return groups
    cluster_means = {label: np.mean(crs_values[labels == label]) for label in unique_labels}
    sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])  # Sort by mean CRS
    label_mapping = {sorted_clusters[0][0]: 'Low', sorted_clusters[1][0]: 'Intermediate', sorted_clusters[2][0]: 'High'}
    return np.array([label_mapping.get(label, 'Intermediate') for label in labels])

def calculate_balance_score(groups):
    unique_labels, counts = np.unique(groups, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0

def compute_cohens_d(param_values, labels_codes):
    cohens_ds = []
    for g1, g2 in combinations(range(len(np.unique(labels_codes))), 2):
        group_values_g1 = param_values[labels_codes == g1]
        group_values_g2 = param_values[labels_codes == g2]
        if len(group_values_g1) > 1 and len(group_values_g2) > 1:
            mean1, mean2 = np.mean(group_values_g1), np.mean(group_values_g2)
            std1, std2 = np.std(group_values_g1, ddof=1), np.std(group_values_g2, ddof=1)
            n1, n2 = len(group_values_g1), len(group_values_g2)
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = abs((mean1 - mean2) / pooled_std) if pooled_std > 0 else 0.0
            cohens_ds.append(cohens_d)
    return np.mean(cohens_ds) if cohens_ds else 0.0

# --- LOOCV for clustering ---
def perform_loocv_clustering(crs_array, model, algo_name, n_min=1, output_dir="CRS_ML_Clustering_LOOCV_Results"):
    Path(output_dir).mkdir(exist_ok=True)
    patient_ids = crs_series.index.to_list()
    loocv_results = []
    consistency_records = []
    left_out_match_flags = []
    skipped_count = 0
    valid_count = 0

    # Original clustering
    model.fit(crs_array)
    original_labels = model.predict(crs_array) if algo_name == 'CRS-GMM' else model.labels_
    original_groups = map_cluster_labels_to_groups(original_labels, crs_series.values)
    original_groups_series = pd.Series(original_groups, index=patient_ids)

    print(f"Starting LOOCV for {algo_name} with {len(patient_ids)} patients...")

    for left_out_id in patient_ids:
        idx = patient_ids.index(left_out_id)
        loo_array = np.delete(crs_array, idx, axis=0)
        loo_ids = [p for i, p in enumerate(patient_ids) if i != idx]

        if len(loo_array) < 3:
            skipped_count += 1
            print(f"Skipped {left_out_id}: Not enough samples ({len(loo_array)})")
            continue

        # Re-fit model on LOO data
        model.fit(loo_array)
        loo_labels = model.predict(loo_array) if algo_name == 'CRS-GMM' else model.labels_
        loo_groups = map_cluster_labels_to_groups(loo_labels, crs_series.loc[loo_ids].values)
        loo_groups_series = pd.Series(loo_groups, index=loo_ids)

        # Check groups (use fallback if needed)
        unique_loo, counts_loo = np.unique(loo_groups, return_counts=True)
        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            # Fallback: use quantile-based grouping for LOO
            quantiles = np.percentile(crs_series.loc[loo_ids].values, [33, 66])
            loo_groups_fallback = np.where(crs_series.loc[loo_ids].values < quantiles[0], 'Low',
                                          np.where(crs_series.loc[loo_ids].values > quantiles[1], 'High', 'Medium'))
            loo_groups_series = pd.Series(loo_groups_fallback, index=loo_ids)
            unique_loo, counts_loo = np.unique(loo_groups_series, return_counts=True)
            print(f"Fallback used for {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")

        if len(unique_loo) != 3 or np.any(counts_loo < n_min):
            skipped_count += 1
            print(f"Skipped {left_out_id}: unique_loo={unique_loo}, counts={counts_loo}")
            continue

        valid_count += 1
        print(f"Processed {left_out_id}: groups={unique_loo}, counts={counts_loo}")

        # Assign label to left-out patient
        left_out_crs = crs_array[idx].reshape(1, -1)  # Reshape for GMM predict
        if algo_name == 'CRS-KMeans':
            distances = np.abs(model.cluster_centers_.flatten() - left_out_crs.flatten()[0])
            loo_label_leftout_idx = np.argmin(distances)
            loo_label_leftout = map_cluster_labels_to_groups(np.array([loo_label_leftout_idx]), np.array([left_out_crs.flatten()[0]]))[0]
        elif algo_name == 'CRS-GMM':
            loo_label_leftout_idx = model.predict(left_out_crs)[0]
            loo_label_leftout = map_cluster_labels_to_groups(np.array([loo_label_leftout_idx]), np.array([left_out_crs.flatten()[0]]))[0]
        else:  # Hierarchical
            # Use nearest neighbor
            distances = np.abs(crs_series.loc[loo_ids].values - left_out_crs.flatten()[0])
            nearest_pid = loo_ids[np.argmin(distances)]
            loo_label_leftout = loo_groups_series.loc[nearest_pid]

        orig_label_leftout = original_groups_series.loc[left_out_id]
        left_out_match = int(loo_label_leftout == orig_label_leftout)
        left_out_match_flags.append(left_out_match)

        # Record remaining patients
        loo_result_df = pd.DataFrame({
            'Patient_ID': loo_ids,
            'CRS': crs_series.loc[loo_ids].values,
            'LOOCV_Group': loo_groups_series.values,
            'Original_Group': original_groups_series.loc[loo_ids].values,
            'Left_Out_ID': left_out_id,
            'Left_Out_Score': left_out_crs.flatten()[0],
            'Left_Out_LOOCV_Group': loo_label_leftout,
            'Left_Out_Original_Group': orig_label_leftout,
            'Left_Out_Match': left_out_match
        })
        loo_result_df['Changed'] = (loo_result_df['LOOCV_Group'] != loo_result_df['Original_Group']).astype(int)
        loo_filename = f"{output_dir}/LOOCV_Result_Patient_{left_out_id}.csv"
        loo_result_df.to_csv(loo_filename, index=False)
        loocv_results.append(loo_result_df)

        # Consistency for remaining patients
        for pid in loo_ids:
            changed_flag = int(original_groups_series.loc[pid] != loo_groups_series.loc[pid])
            consistency_records.append(changed_flag)

    print(f"LOOCV for {algo_name}: Valid iterations = {valid_count}, Skipped = {skipped_count}")
    mean_change_rate = float(np.mean(consistency_records)) if consistency_records else np.nan
    mean_match_rate = float(np.mean(left_out_match_flags)) if left_out_match_flags else np.nan
    loocv_results_df = pd.concat(loocv_results, ignore_index=True) if loocv_results else pd.DataFrame()
    return mean_change_rate, mean_match_rate, loocv_results_df

# --- Run for each algorithm ---
results_summary = []
for (algo_name, model), out_dir in zip(algorithms, output_dirs):
    print(f"\n--- Running {algo_name} Clustering ---")
    Path(out_dir).mkdir(exist_ok=True)

    # Fit model on full data
    model.fit(crs_array)
    cluster_labels = model.predict(crs_array) if algo_name == 'CRS-GMM' else model.labels_
    if len(np.unique(cluster_labels)) != 3:
        print(f"Warning: {algo_name} produced {len(np.unique(cluster_labels))} clusters, using fallback mapping.")
        # Use fallback mapping
        quantiles = np.percentile(crs_series.values, [33, 66])
        cluster_labels = np.where(crs_series.values < quantiles[0], 0,
                                 np.where(crs_series.values > quantiles[1], 2, 1))
    groups = map_cluster_labels_to_groups(cluster_labels, crs_series.values)
    df[f'{algo_name}_Cluster'] = groups

    # Compute metrics
    labels_codes = pd.Series(groups).astype('category').cat.codes
    sil_curve = silhouette_score(curve_features_df.values, labels_codes)
    db_curve = davies_bouldin_score(curve_features_df.values, labels_codes)

    # Cohen's d
    param_features = df.loc[crs_series.index, ['MED', 'IC50', 'IC90']].copy()
    param_features = np.log1p(param_features)  # Log-transform to stabilize variance
    cohens_ds = []
    for param in ['MED', 'IC50', 'IC90']:
        param_values = param_features[param].values
        group_values = [param_values[labels_codes == i] for i in range(3)]
        for g1, g2 in combinations(range(3), 2):
            std1, std2 = np.std(group_values[g1], ddof=1), np.std(group_values[g2], ddof=1)
            n1, n2 = len(group_values[g1]), len(group_values[g2])
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)) if n1+n2>2 else 0
            cohens_d = (np.mean(group_values[g1]) - np.mean(group_values[g2])) / pooled_std if pooled_std>0 else 0.0
            cohens_ds.append(abs(cohens_d))
    effect_size_curve = np.mean(cohens_ds) if cohens_ds else 0.0

    # Balance
    def calculate_balance_score(groups):
        unique_labels, counts = np.unique(groups, return_counts=True)
        if len(counts) <= 1:
            return 0.0
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    balance = calculate_balance_score(groups)

    # LOOCV
    mean_change_rate, mean_match_rate, loocv_results_df = perform_loocv_clustering(crs_array, model, algo_name, n_min=1, output_dir=out_dir)

    effective_change_rate = 1.0 - mean_change_rate if not np.isnan(mean_change_rate) else np.nan

    # Save metrics
    metrics_row = {
        'Algorithm': algo_name,
        'Silhouette_Curve': sil_curve,
        'Davies_Bouldin_Curve': db_curve,
        'Cohen_d_Curve': effect_size_curve,
        'Balance_Score': balance,
        'LOOCV_Mean_Change_Rate': mean_change_rate,
        'LOOCV_Mean_Match_Rate': mean_match_rate,
        'Effective_Change_Rate': effective_change_rate
    }
    results_summary.append(metrics_row)
    pd.DataFrame([metrics_row]).to_csv(f"{algo_name}_Clustering_Metrics.csv", index=False)
    print(f"Saved metrics to {algo_name}_Clustering_Metrics.csv")

    # Save classification
    df.to_csv(f"{algo_name}_Clustering_Results.csv")
    print(f"Saved classification results to {algo_name}_Clustering_Results.csv")

    # Save LOOCV all results
    loocv_results_df.to_csv(f"{out_dir}/LOOCV_All_Results.csv", index=False)
    print(f"Saved LOOCV results to {out_dir}/LOOCV_All_Results.csv")

    # Save LOOCV summary
    summary_df = pd.DataFrame({
        'Metric': ['Mean_Change_Rate', 'Mean_Match_Rate'],
        'Value': [mean_change_rate, mean_match_rate]
    })
    summary_df.to_csv(f"{out_dir}/LOOCV_Summary.csv", index=False)
    print(f"Saved LOOCV summary to {out_dir}/LOOCV_Summary.csv")

    # Cluster centers statistics (if applicable)
    if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
        cluster_centers = model.means_ if algo_name == 'CRS-GMM' else model.cluster_centers_
        cluster_centers_stats = pd.DataFrame({
            'Cluster': ['Cluster_0', 'Cluster_1', 'Cluster_2'],
            'Center_Mean': cluster_centers.flatten()
        })
        cluster_centers_stats.to_csv(f"{out_dir}/Cluster_Centers_Statistics.csv", index=False)
        print(f"Saved cluster centers to {out_dir}/Cluster_Centers_Statistics.csv")
    else:
        # Fallback for Hierarchical
        cluster_centers_stats = pd.DataFrame({
            'Cluster': ['Cluster_0', 'Cluster_1', 'Cluster_2'],
            'Center_Mean': [np.mean(crs_series[groups == g]) for g in ['Low', 'Intermediate', 'High']]
        })
        cluster_centers_stats.to_csv(f"{out_dir}/Cluster_Centers_Statistics.csv", index=False)
        print(f"Saved cluster centers (fallback) to {out_dir}/Cluster_Centers_Statistics.csv")

    # Left-out summary
    left_out_summary = loocv_results_df[['Left_Out_ID', 'Left_Out_Score', 'Left_Out_LOOCV_Group',
                                        'Left_Out_Original_Group', 'Left_Out_Match']].drop_duplicates()
    left_out_summary.to_csv(f"{out_dir}/Left_Out_Classification_Summary.csv", index=False)
    print(f"Saved left-out summary to {out_dir}/Left_Out_Classification_Summary.csv")

    print(f"✅ {algo_name} clustering and LOOCV results saved in {out_dir}/")

# Save overall summary
overall_summary = pd.DataFrame(results_summary)
overall_summary.to_csv("CRS_ML_Clustering_Overall_Metrics.csv", index=False)
print("\nOverall clustering metrics saved to CRS_ML_Clustering_Overall_Metrics.csv")

# --- Visualization for each algorithm ---
for (algo_name, model), out_dir in zip(algorithms, output_dirs):
    if f'{algo_name}_Cluster' not in df.columns:
        continue  # Skip if not processed

    sns.set(style="whitegrid")
    groups = df[f'{algo_name}_Cluster']
    crs_values = crs_series

    # Classification Visualization
    plt.figure(figsize=(20, 6))
    plt.suptitle(f'{algo_name} Clustering Analysis (CRS)', fontsize=14, y=1.02)

    # 1. CRS distribution
    plt.subplot(1, 3, 1)
    sns.kdeplot(crs_values, fill=True, color='#4e79a7', alpha=0.5)
    # Cluster centers as 'thresholds'
    if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
        centers = model.means_ if algo_name == 'CRS-GMM' else model.cluster_centers_
        centers = centers.flatten()
        plt.axvline(np.min(centers), color='g', linestyle='--', label='Cluster Centers Min')
        plt.axvline(np.max(centers), color='r', linestyle='--', label='Cluster Centers Max')
    plt.title('CRS Distribution', fontsize=12)
    plt.xlabel('CRS')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.2)

    # 2. Patient grouping scatter plot
    plt.subplot(1, 3, 2)
    crs_output_df = df[['CRS', f'{algo_name}_Cluster']].copy()
    crs_output_df.columns = ['CRS', 'Cluster']
    group_colors = {'Low': '#e15759', 'Intermediate': '#f28e2b', 'High': '#4e79a7'}
    for group, color in group_colors.items():
        mask = crs_output_df['Cluster'] == group
        plt.scatter(crs_output_df.index[mask], crs_output_df.loc[mask, 'CRS'],
                    label=group, c=color, alpha=0.7, s=50)
    plt.title(f'Patient Grouping by {algo_name}', fontsize=12)
    plt.xlabel('Patient ID')
    plt.ylabel('CRS')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)

    # 3. Group box plot
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Cluster', y='CRS', data=crs_output_df,
                order=['Low', 'Intermediate', 'High'],
                palette=['#e15759', '#f28e2b', '#4e79a7'])
    plt.title('CRS Distribution by Cluster', fontsize=12)
    plt.xlabel('Sensitivity Group')
    plt.ylabel('CRS')
    plt.grid(alpha=0.2)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{algo_name}_Clustering_Visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

    # LOOCV Visualization
    loocv_df = pd.read_csv(f"{out_dir}/LOOCV_All_Results.csv")
    if not loocv_df.empty:
        plt.figure(figsize=(20, 12))
        plt.suptitle(f'{algo_name} LOOCV Analysis (CRS)', fontsize=16, y=1.02)

        # 1. CRS mean variation plot
        plt.subplot(2, 2, 1)
        threshold_stats_vis = loocv_df.groupby('Left_Out_ID').agg({
            'CRS': 'mean'
        }).reset_index()
        plt.plot(threshold_stats_vis['Left_Out_ID'], threshold_stats_vis['CRS'], label='CRS Mean', color='#4e79a7')
        plt.title('CRS Mean Variation Across LOOCV', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('CRS Value')
        plt.legend()
        plt.grid(alpha=0.2)

        # 2. Classification change heatmap
        pivot_table = loocv_df.pivot_table(index='Patient_ID', columns='Left_Out_ID', values='Changed', aggfunc='mean')
        plt.subplot(2, 2, 2)
        sns.heatmap(pivot_table, cmap='RdYlGn_r', cbar_kws={'label': 'Change Magnitude (0=No, 1=Yes)'}, annot=True, fmt='.2f')
        plt.title('Classification Change Heatmap', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('Patient ID')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 3. LOOCV group distribution
        plt.subplot(2, 2, 3)
        sns.countplot(data=loocv_df, x='LOOCV_Group', hue='Original_Group',
                      palette=['#e15759', '#f28e2b', '#4e79a7'])
        plt.title('LOOCV Group vs Original Group Distribution', fontsize=12)
        plt.xlabel('LOOCV Group')
        plt.ylabel('Count')
        plt.legend(title='Original Group')
        plt.grid(alpha=0.2)

        # 4. Left-out patient match rate
        left_out_summary_vis = loocv_df[['Left_Out_ID', 'Left_Out_Match']].groupby('Left_Out_ID')['Left_Out_Match'].mean().reset_index()
        plt.subplot(2, 2, 4)
        sns.barplot(data=left_out_summary_vis, x='Left_Out_ID', y='Left_Out_Match', color='#4e79a7')
        plt.title('Left-Out Patient Match Rate Across LOOCV', fontsize=12)
        plt.xlabel('Left-Out Patient ID')
        plt.ylabel('Match Rate')
        plt.grid(alpha=0.2)
        plt.xticks(rotation=45)

        plt.tight_layout(pad=2.0)
        plt.savefig(f"{out_dir}/{algo_name}_LOOCV_Analysis_Visualization.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Boxplot for parameters
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{algo_name} Parameter Statistics Across LOOCV', fontsize=14, y=1.02)

        plt.subplot(1, 2, 1)
        sns.boxplot(data=loocv_df[['CRS']], palette=['#4e79a7'])
        plt.title('CRS Distribution', fontsize=12)
        plt.ylabel('CRS Value')
        plt.grid(alpha=0.2)

        plt.subplot(1, 2, 2)
        if hasattr(model, 'means_') or hasattr(model, 'cluster_centers_'):
            centers = model.means_ if algo_name == 'CRS-GMM' else model.cluster_centers_
            centers_df = pd.DataFrame({'Cluster_Centers': centers.flatten()})
            sns.boxplot(data=centers_df, palette=['#4e79a7'])
            plt.title('Cluster Centers Distribution', fontsize=12)
            plt.ylabel('Center Value')
        else:
            plt.text(0.5, 0.5, 'No cluster centers available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Cluster Centers (N/A)', fontsize=12)
        plt.grid(alpha=0.2)

        plt.tight_layout(pad=2.0)
        plt.savefig(f"{out_dir}/Parameter_Boxplot.png", dpi=300, bbox_inches='tight')
        plt.show()

print("\n===== Step 10 CRS Clustering with ML Algorithms and LOOCV Completed =====")






# ======================
# Step 11. Comparison of Nine Classification Methods
# ======================

# --- Define methods and their metrics file paths ---
methods = [
    ('MSSCF', 'CRS_ASI_Classification_Metrics.csv'),
    ('IC50_CI', 'IC50_CI_Classification_Metrics.csv'),
    ('CRS_CI', 'CRS_CI_Classification_Metrics.csv'),
    ('IC50_KMeans', 'IC50-KMeans_Clustering_Metrics.csv'),
    ('IC50_Hierarchical', 'IC50-Hierarchical_Clustering_Metrics.csv'),
    ('IC50_GMM', 'IC50-GMM_Clustering_Metrics.csv'),
    ('CRS_KMeans', 'CRS-KMeans_Clustering_Metrics.csv'),
    ('CRS_Hierarchical', 'CRS-Hierarchical_Clustering_Metrics.csv'),
    ('CRS_GMM', 'CRS-GMM_Clustering_Metrics.csv'),
]

# --- Load and combine metrics ---
metrics_list = []
for method_name, file_path in methods:
    try:
        df_metrics = pd.read_csv(file_path)
        # 确保 Algorithm 列存在
        if 'Algorithm' not in df_metrics.columns:
            df_metrics['Algorithm'] = method_name
            print(f"Added 'Algorithm' column with value '{method_name}' for {file_path}")
        metrics_list.append(df_metrics)
        print(f"Loaded metrics for {method_name} from {file_path}")
    except FileNotFoundError:
        print(f"Warning: Metrics file {file_path} not found, skipping {method_name}")

if not metrics_list:
    raise FileNotFoundError("No metrics files were loaded. Please check file paths and ensure previous steps have run successfully.")

combined_metrics = pd.concat(metrics_list, ignore_index=True)

# --- Ensure all expected columns are present ---
expected_columns = [
    'Algorithm', 'Silhouette_Curve', 'Davies_Bouldin_Curve', 'Cohen_d_Curve',
    'Balance_Score', 'LOOCV_Mean_Change_Rate', 'LOOCV_Mean_Match_Rate', 'Effective_Change_Rate'
]
for col in expected_columns:
    if col not in combined_metrics.columns:
        combined_metrics[col] = np.nan
        print(f"Warning: Column {col} missing in some metrics files, filled with NaN")

# 检查缺失值
print(f"\nMissing values in combined_metrics:\n{combined_metrics[expected_columns].isna().sum()}")

# 检查指标分布
print("\nMetrics summary:")
for col in expected_columns[1:]:  # 跳过 Algorithm
    print(f"{col}:\n{combined_metrics[col].describe()}")

# Rename 'Algorithm' to 'Method' for consistency
combined_metrics = combined_metrics.rename(columns={'Algorithm': 'Method'})

# --- Calculate composite scores ---
# 1. Standardize metrics and compute CSS (Clustering Separation Score)
css_metrics = ['Silhouette_Curve', 'Davies_Bouldin_Curve', 'Cohen_d_Curve']
for col in css_metrics:
    min_val = combined_metrics[col].min()
    max_val = combined_metrics[col].max()
    if max_val > min_val:
        if col == 'Davies_Bouldin_Curve':
            # Davies_Bouldin_Curve is lower-is-better, so invert normalization
            combined_metrics[f'Norm_{col}'] = (max_val - combined_metrics[col]) / (max_val - min_val)
        else:
            combined_metrics[f'Norm_{col}'] = (combined_metrics[col] - min_val) / (max_val - min_val)
    else:
        combined_metrics[f'Norm_{col}'] = 1.0  # If all values are same, assume best performance
        print(f"Warning: {col} has no variation (min = max = {min_val}), set Norm_{col} to 1.0")
combined_metrics['CSS'] = combined_metrics[[f'Norm_{col}' for col in css_metrics]].mean(axis=1)

# 2. Standardize Balance_Score to CBS (Clustering Balance Score)
min_val = combined_metrics['Balance_Score'].min()
max_val = combined_metrics['Balance_Score'].max()
if max_val > min_val:
    combined_metrics['CBS'] = (combined_metrics['Balance_Score'] - min_val) / (max_val - min_val)
else:
    combined_metrics['CBS'] = 1.0
    print(f"Warning: Balance_Score has no variation (min = max = {min_val}), set CBS to 1.0")

# 3. Standardize LOOCV metrics and compute CRS (Clustering Robustness Score)
crs_metrics = ['LOOCV_Mean_Match_Rate', 'Effective_Change_Rate']
for col in crs_metrics:
    min_val = combined_metrics[col].min()
    max_val = combined_metrics[col].max()
    if max_val > min_val:
        combined_metrics[f'Norm_{col}'] = (combined_metrics[col] - min_val) / (max_val - min_val)
    else:
        combined_metrics[f'Norm_{col}'] = 1.0
        print(f"Warning: {col} has no variation (min = max = {min_val}), set Norm_{col} to 1.0")
combined_metrics['CRS'] = combined_metrics[[f'Norm_{col}' for col in crs_metrics]].mean(axis=1)

# 4. Compute ICES (Integrated Clustering Evaluation Score)
combined_metrics['ICES'] = (combined_metrics['CSS'] + combined_metrics['CBS'] + combined_metrics['CRS']) / 3

# 5. Sort by ICES in descending order
combined_metrics = combined_metrics.sort_values(by='ICES', ascending=False)

# --- Save combined metrics ---
combined_metrics.to_csv(f"Methods_Comparison_Metrics.csv", index=False)
print(f"\nCombined metrics saved to Methods_Comparison_Metrics.csv")

# --- Print ranking ---
print("\nRanking of methods by ICES (descending order):")
print(combined_metrics[['Method', 'ICES']].to_string(index=False))

# --- Visualization ---
sns.set(style="whitegrid")

# 1. Barplot for original and composite metrics
metrics_to_plot = [
    'Silhouette_Curve', 'Davies_Bouldin_Curve', 'Cohen_d_Curve',
    'Balance_Score', 'LOOCV_Mean_Match_Rate', 'Effective_Change_Rate',
    'CSS', 'CBS', 'CRS', 'ICES'
]
plt.figure(figsize=(20, 12))
plt.suptitle('CDDP Comparison of Clustering Methods Across Metrics', fontsize=16, y=1.02)

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(4, 3, i)
    sns.barplot(data=combined_metrics, x='Method', y=metric, hue='Method', palette='tab10', legend=False)
    plt.title(metric, fontsize=12)
    plt.xlabel('Method')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.2)

plt.tight_layout(pad=2.0)
plt.savefig(f"Comparison_Barplot.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. ICES-specific barplot (sorted by ICES)
plt.figure(figsize=(10, 6))
sns.barplot(data=combined_metrics, x='Method', y='ICES', hue='Method', palette='tab10', legend=False)
plt.title('Ranking of Clustering Methods by ICES', fontsize=14)
plt.xlabel('Method')
plt.ylabel('ICES')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"ICES_Ranking_Barplot.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Heatmap of normalized metrics
normalized_metrics = combined_metrics[[
    'Norm_Silhouette_Curve', 'Norm_Davies_Bouldin_Curve', 'Norm_Cohen_d_Curve',
    'CBS', 'Norm_LOOCV_Mean_Match_Rate', 'Norm_Effective_Change_Rate', 'CSS', 'CRS', 'ICES'
]].copy()
normalized_metrics['Method'] = combined_metrics['Method']
plt.figure(figsize=(12, 8))
sns.heatmap(normalized_metrics.set_index('Method'), annot=True, cmap='YlGnBu', cbar_kws={'label': 'Normalized Value'})
plt.title('Normalized Metrics Across Clustering Methods', fontsize=14)
plt.xlabel('Metric')
plt.ylabel('Method')
plt.tight_layout()
plt.savefig(f"Comparison_Heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Radar plot for composite scores
def plot_radar(data, categories, methods, title, output_file):
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for method in methods:
        values = data[data['Method'] == method][categories].values.flatten().tolist()
        values += values[:1]  # Close the circle
        ax.plot(angles, values, label=method, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

plot_radar(
    combined_metrics,
    ['CSS', 'CBS', 'CRS', 'ICES'],
    combined_metrics['Method'].unique(),
    'CDDP Radar Plot of Composite Metrics Across Methods',
     f"Composite_Metrics_Radarplot.png"
)

print("\n===== Step 11 Comparison of Nine Classification Methods Completed =====")





