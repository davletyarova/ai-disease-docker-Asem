import pandas as pd
import pickle
from predictor.symptoms import SYMPTOMS
from predictor.descriptions import DISEASE_DESCRIPTIONS, DISEASE_RECOMMENDATIONS

# === 1. Проверка симптомов ===
df = pd.read_csv('dataset/Training.csv')
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df_symptoms = set(df.columns) - {'prognosis'}

# Найти симптомы, которых нет в датасете
missing_in_dataset = [s for s in SYMPTOMS if s not in df_symptoms]
extra_in_dataset = [s for s in df_symptoms if s not in SYMPTOMS]

print("❌ Симптомы из SYMPTOMS, которых нет в CSV:", missing_in_dataset)
print("⚠️ Симптомы в CSV, которых нет в SYMPTOMS:", extra_in_dataset)   

# === 2. Проверка болезней ===
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

diseases = list(le.classes_)
missing_desc = [d for d in diseases if d not in DISEASE_DESCRIPTIONS]
missing_reco = [d for d in diseases if d not in DISEASE_RECOMMENDATIONS]

print("\n📄 Описания не найдены для:", missing_desc)
print("📄 Рекомендации не найдены для:", missing_reco)
