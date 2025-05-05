from scipy.special import expit  # sigmoid
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def generate_data(n_per_class=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

# --- University definitions (difficulty on a 0–10 scale) ---
    universities = {
        'FAST':    {'difficulty': 7, 'fields': ['CS']},
        'NUST':    {'difficulty': 8, 'fields': ['CS', 'Engineering', 'Medicine']},
        'UET':     {'difficulty': 6, 'fields': ['Engineering']},
        'COMSAT': {'difficulty': 5, 'fields': ['CS', 'Business']},
        'GIKI':    {'difficulty': 9, 'fields': ['Engineering', 'CS']}
    }

    fields  = ['CS', 'Engineering', 'Medicine', 'Business']
    regions = ['GB', 'Punjab', 'Sindh', 'KPK', 'Balochistan']

    # how much each subject “counts” for each field
    field_weights = {
        'CS':          {'math': 0.50, 'physics': 0.30, 'english': 0.20},
        'Engineering': {'math': 0.40, 'physics': 0.40, 'english': 0.20},
        'Medicine':    {'math': 0.30, 'physics': 0.40, 'english': 0.30},
        'Business':    {'math': 0.35, 'physics': 0.25, 'english': 0.40},
    }

    def calculate_weighted_average(field, math, physics, english):
        w = field_weights[field]
        return math*w['math'] + physics*w['physics'] + english*w['english']

    def student_field_tendency(field):
        # baseline strengths
        return {
            'CS':          {'math': 85, 'physics': 35, 'english': 20},
            'Engineering': {'math': 80, 'physics': 50, 'english': 15},
            'Medicine':    {'math': 65, 'physics': 45, 'english': 32},
            'Business':    {'math': 55, 'physics': 30, 'english': 35},
        }[field]

    def generate_marks(base, boost, max_score):
        """Truncated normal around (base+boost), clipped [35%*max, max]."""
        lower, upper = 0.35*max_score, max_score
        mean, sd = base+boost, 10
        a, b = (lower-mean)/sd, (upper-mean)/sd
        return round(truncnorm.rvs(a, b, loc=mean, scale=sd), 1)

    def generate_time_pct():
        """
        Returns a fraction of allotted time: N(1.0, 0.1) clipped [0.4,1.0].
        """
        return float(np.clip(np.random.normal(1.0, 0.1), 0.4, 1.0))

    # Admission‐prob formula
    def calculate_p(avg, diff, attempts):
        base     = expit((avg - 60) * 0.04)
        diff_fac = 1 - diff/10
        boost    = attempts * 0.05
        raw      = base * (diff_fac + 0.6) + boost
        return float(np.clip(raw, 0.1, 0.95))



    # Targets & counters
    target_yes= n_per_class
    target_no = n_per_class
    yes_count = 0
    no_count = 0
    admit_ctr = {uni: 0 for uni in universities}
    
    MAX_PER_UNI = n_per_class/5

    data = []
    student_idx = 0
    iterations = 0

    while (yes_count < target_yes or no_count < target_no):
        iterations += 1

        # 1. Pick field & region
        field  = random.choice(fields)
        region = random.choice(regions)

        # 2. Simulate marks & avg
        base_tend = student_field_tendency(field)
        attempts  = random.randint(1,3)
        boost     = (attempts-1)*3

        phys = generate_marks(base_tend['physics'], boost, 60)
        math = generate_marks(base_tend['math'],     boost,100)
        eng  = generate_marks(base_tend['english'],  boost, 40)
        avg  = calculate_weighted_average(field, math, phys, eng)

        # 3. Which unis can they apply to?
        possibles = [
        u for u,info in universities.items()
        if field in info['fields'] and admit_ctr[u] < MAX_PER_UNI
        ]
        # if not possibles:
        #         print(">>>>>>>>>>>>>>>>>>>>>>>>here")
        #         print(f"Iteration: {iterations}, Yes: {yes_count}, No: {no_count}, Admit Counts: {admit_ctr}")
        #         continue

        choices = random.sample(possibles, k=min(3, len(possibles)))

        # 4. Do the admission draws
        passed = {}
        for u in choices:
            p = calculate_p(avg, universities[u]['difficulty'], attempts)
            if random.random() < p:
                passed[u] = p

        if passed:
            admit_uni = max(passed, key=passed.get)
            yes_count += 1
            admit_ctr[admit_uni] += 1
            status = 1
        elif no_count < target_no:
            no_count += 1
            admit_uni = 'None'
            status = 0
        else:
            print("<<<<<<<<here")
            continue

        # 5. Record & iterate
        student_idx += 1
        data.append({
            'student_id':                 f"ST{student_idx:05}",
            'physics_marks':              phys,
            'math_marks':                 math,
            'english_marks':              eng,
            'average_marks':              round(avg, 2),
            'time_physics_pct':           generate_time_pct(),
            'time_math_pct':              generate_time_pct(),
            'time_english_pct':           generate_time_pct(),
            'number_of_attempts':         attempts,
            'preferred_field':            field,            # <-- Added here
            'region_preference':          region,
            'university_choices':         choices,
            'admission_status':           status,
            'admitted_university':        admit_uni,
            'university_difficulty_index': universities[admit_uni]['difficulty'] if admit_uni!='None' else None
        })

        if (yes_count + no_count) % 1000 == 0:
            print(f"Yes:{yes_count}  No:{no_count}  Total:{yes_count+no_count}")

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    base_threshold = 50
    for univ, idx in universities.items():
        df[f'perf_gap_{univ}'] = df['average_marks'] - base_threshold * idx['difficulty']
    df['phy_norm']  = df['physics_marks'] / 60
    df['math_norm'] = df['math_marks'] / 100
    df['eng_norm']  = df['english_marks'] / 40
    df[['choice1','choice2','choice3']] = pd.DataFrame(df['university_choices'].tolist(), index=df.index)
    df['eff_phy'] = df['time_physics_pct'] * df['phy_norm']
    df['eff_math'] = df['time_math_pct'] * df['math_norm']
    df['eff_eng'] = df['time_english_pct'] * df['eng_norm']
    df['attempt_x_diff'] = df['number_of_attempts'] * df['university_difficulty_index'].fillna(1)

    return df
# df.to_csv('realistic_student_data.csv', index=False)


# 1. Regenerate synthetic data (smaller test set)

# Prepare training pipelines and save
df= generate_data(n_per_class=20000)
df['admission_status']=df['admission_status'].map({'Yes':1,'No':0})
df_train=df
# Define preprocessor
numeric_feats = ['average_marks','phy_norm','math_norm','eng_norm','time_physics_pct',
                 'time_math_pct','time_english_pct','number_of_attempts','attempt_x_diff'] + \
                [f'perf_gap_{u}' for u in ['FAST','NUST','GIKI','UET','COMSAT']]
categorical_feats = ['preferred_field','region_preference','choice1','choice2','choice3']

master_ct = ColumnTransformer([
    ('num', StandardScaler(),      numeric_feats),
    ('cat', OneHotEncoder(
                sparse_output=False,
                drop='first',
                handle_unknown='ignore'
            ), categorical_feats),
])

# 2) clone for each stage
ct1 = clone(master_ct)   # fresh un‑fitted copy
ct2 = clone(master_ct)   # another fresh copy


# Stage 1 pipeline
pipe1 = Pipeline([
    ('prep', ct1),
    ('clf', RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,            # limit tree depth
    min_samples_leaf=100,      # require at least 5 samples to form a leaf
    max_features='sqrt',     # limit how many features each split considers
    oob_score=True,        # enable out-of-bag scoring for a built-in generalization check
    class_weight="balanced",
    random_state=42,
    n_jobs = -1))
])
X1 = df_train.drop(columns=['admission_status','admitted_university','university_difficulty_index','university_choices'])
y1 = df_train['admission_status']
pipe1.fit(X1, y1)

# Stage 2 pipeline (only on admitted)
df_adm = df[df['admission_status']==1]
pipe2 = Pipeline([
    ('prep', ct2),
    ('clf', RandomForestClassifier(    
    n_estimators=1000,
    max_depth=10,            # limit tree depth
    min_samples_leaf=100,      # require at least 5 samples to form a leaf
    max_features='sqrt',     # limit how many features each split considers
    oob_score=True,        # enable out-of-bag scoring for a built-in generalization check
    class_weight="balanced",
    random_state=0,
     n_jobs = -1))
])
X2 = df_adm.drop(columns=['admission_status','admitted_university','university_difficulty_index','university_choices'])
y2 = df_adm['admitted_university']
pipe2.fit(X2, y2)

# Save pipelines
joblib.dump(pipe1, 'stage1_pipeline.pkl')
joblib.dump(pipe2, 'stage2_pipeline.pkl')

# Inference function
def predict_admission(df_raw):
    # df_raw must have same raw columns as df_train
    df_raw['admission_status']=df_raw['admission_status'].map({'Yes':1,'No':0})
    X = df_raw.copy()
    # Stage 1
    adm_pred = pipe1.predict(X)
    df_raw['pred_admission_status'] = adm_pred
    # Stage 2 for those admitted
    mask = df_raw['pred_admission_status'] == 1
    df_raw.loc[mask, 'pred_university'] = pipe2.predict(X[mask])

    # Safe fill without inplace
    df_raw['pred_university'] = df_raw['pred_university'].fillna('None')
    return df_raw

# 2. Generate random test data
df_test = generate_data(n_per_class=20000, seed=123)
df_results = predict_admission(df_test)

# Evaluate
print("=== Stage 1 Evaluation ===")
print(classification_report(df_test['admission_status'], df_results['pred_admission_status'], zero_division=0))

print("\n=== Stage 2 Evaluation (Admitted Only) ===")
mask = df_test['admission_status']==1
print(classification_report(df_test.loc[mask,'admitted_university'], df_results.loc[mask,'pred_university'], zero_division=0))

# Sample predictions
print("\nSample predictions:")
print(df_results[['physics_marks','math_marks','english_marks','admission_status','pred_admission_status','admitted_university','pred_university']].head())




# 1) Load your models
pipe1 = joblib.load("stage1_pipeline.pkl")
pipe2 = joblib.load("stage2_pipeline.pkl")

# 2) Define a few hand‑crafted students
cases = [
    # definitely too weak → no admission
    {"physics_marks": 30, "math_marks": 60, "english_marks": 20, 
     "time_physics_pct": .9, "time_math_pct": .1, "time_english_pct": .2,
     "number_of_attempts": 1, "preferred_field": "Business", "region_preference": "Rural",
     "university_choices": ["FAST","GIKI","None"]},
    # stellar on FAST’s scale → FAST
    {"physics_marks": 45, "math_marks": 50, "english_marks": 35,
     "time_physics_pct": .3, "time_math_pct": .5, "time_english_pct": .2,
     "number_of_attempts": 1, "preferred_field": "Engineering", "region_preference": "Urban",
     "university_choices": ["FAST","COMSAT"]},
    # mid‑range but ranks NUST first → NUST
    {"physics_marks": 60, "math_marks": 95, "english_marks": 30,
     "time_physics_pct": .1, "time_math_pct": .2, "time_english_pct": .2,
     "number_of_attempts": 2, "preferred_field": "Engineering", "region_preference": "Rural",
     "university_choices": ["NUST","GIKI","FAST"]},
]

def engineer(df):
    
    difficulty_index = {
        'FAST': 7.00,
        'NUST': 8.00,
        'GIKI': 9.00,
        'UET':  6.00,
        'COMSAT': 5.00
    }
    # 1. average_marks (however you defined it in training—if it was random, you’ll need a deterministic proxy here)
    df = df.copy()
    df['average_marks'] = (df['physics_marks'] + df['math_marks'] + df['english_marks']) / 3

    # 2. difficulty index lookup
   
    # 3. performance gaps
    base_threshold = 50
    for univ, idx in universities.items():
        df[f'perf_gap_{univ}'] = df['average_marks'] - base_threshold * idx['difficulty']

    # 4. normalized marks
    df['phy_norm']  = df['physics_marks'] / 60
    df['math_norm'] = df['math_marks']    /100
    df['eng_norm']  = df['english_marks'] / 40

    # 5. split ranked choices into separate columns
    df[['choice1','choice2','choice3']] = pd.DataFrame(df['university_choices'].tolist(),
                                                      index=df.index)
    df['eff_phy']  = df['time_physics_pct'] * df['phy_norm']
    df['eff_math'] = df['time_math_pct']    * df['math_norm']
    df['eff_eng']  = df['time_english_pct'] * df['eng_norm']

    # 6) map each student’s first choice to its difficulty index
    df['university_difficulty_index'] = df['choice1'].map(difficulty_index)

    # 7) multiply attempts by that index
    df['attempt_x_diff'] = df['number_of_attempts'] * df['university_difficulty_index']

    return df

# then in your test script:
df_raw = pd.DataFrame(cases)
df_test = engineer(df_raw)





admissions = pipe1.predict(df_test)            # array of 0/1
mask       = admissions == 1
assigned   = np.array(["None"] * len(admissions), dtype=object)
assigned[mask] = pipe2.predict(df_test.loc[mask])

# 2) Display inputs → outputs
for i, case in enumerate(cases):
    print(f"Case {i+1}:")
    print("  Input:", case)
    print(f"  → Admission prediction:       {admissions[i]}")
    print(f"  → Assigned university (if any): {assigned[i]}")
    print("-" * 60)