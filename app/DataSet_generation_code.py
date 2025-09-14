from scipy.special import expit  # sigmoid
import pandas as pd
import numpy as np
import random
from scipy.stats import truncnorm

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
        # baseline strengths according to fields
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


        choices = random.sample(possibles, k=min(3, len(possibles)))

       
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
            'preferred_field':            field,           
            'region_preference':          region,
            'university_choices':         choices,
            'admission_status':           status,
            'admitted_university':        admit_uni,
            'university_difficulty_index': universities[admit_uni]['difficulty'] if admit_uni!='None' else None
        })

        if (yes_count + no_count) % 1000 == 0:
            print(f"Yes:{yes_count}  No:{no_count}  Total:{yes_count+no_count}")


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
