from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = BASE_DIR / 'models'

TRAIN_CSV = DATASET_DIR / 'train.csv'
UNIQUE_M_CSV = DATASET_DIR / 'unique_m.csv'

ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn'
]

FEATURE_COLUMNS = [
    'number_of_elements',
    'mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 'wtd_gmean_atomic_mass',
    'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass',
    'std_atomic_mass', 'wtd_std_atomic_mass',
    'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie',
    'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie',
    'std_fie', 'wtd_std_fie',
    'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 'wtd_gmean_atomic_radius',
    'entropy_atomic_radius', 'wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius',
    'std_atomic_radius', 'wtd_std_atomic_radius',
    'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'wtd_gmean_Density',
    'entropy_Density', 'wtd_entropy_Density', 'range_Density', 'wtd_range_Density',
    'std_Density', 'wtd_std_Density',
    'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
    'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
    'std_ElectronAffinity', 'wtd_std_ElectronAffinity',
    'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat',
    'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat',
    'std_FusionHeat', 'wtd_std_FusionHeat',
    'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
    'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
    'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
    'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence',
    'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence',
    'std_Valence', 'wtd_std_Valence'
]

TARGET_COLUMN = 'critical_temp'
FORMULA_COLUMN = 'material'

LINEAR_MODEL_PATH = MODEL_DIR / 'linear_model.joblib'
RIDGE_MODEL_PATH = MODEL_DIR / 'ridge_model.joblib'
RF_FEATURE_MODEL_PATH = MODEL_DIR / 'rf_feature_model.joblib'
RF_FEATURE_METADATA_PATH = MODEL_DIR / 'rf_feature_model_metadata.json'
RF_FORMULA_MODEL_PATH = MODEL_DIR / 'rf_formula_model.joblib'
SCALER_PATH = MODEL_DIR / 'feature_scaler.joblib'
METRICS_JSON = OUTPUT_DIR / 'metrics_summary.json'

RANDOM_STATE = 42
CV_RANDOM_STATE = 100000
