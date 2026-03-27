import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def prepare_ml_data(fact_lms, fact_activity):
    """
    Merges fact data into a single DataFrame per Intern for ML training.
    """
    if fact_lms.empty or fact_activity.empty:
        return pd.DataFrame()

    # Base aggregation dict
    agg_dict = {
        'Progress_Numeric': 'mean',
        'Reviewed': 'sum',
        'Total_Assg': 'sum'
    }
    
    # Conditionally add to aggregation dict if columns exist
    if 'KC_pct' in fact_lms.columns:
        agg_dict['KC_pct'] = 'mean'
    if 'Test_pct' in fact_lms.columns:
        agg_dict['Test_pct'] = 'mean'

    # Aggregate LMS Data
    lms_agg = fact_lms.groupby('Intern_ID').agg(agg_dict).reset_index()

    # Calculate Assignment Completion Ratio
    lms_agg['Total_Assg'] = lms_agg['Total_Assg'].replace(0, 1)
    lms_agg['Assg_Ratio'] = lms_agg['Reviewed'] / lms_agg['Total_Assg']
    
    # If KC_pct or Test_pct weren't aggregated properly, attempt to parse them
    if 'KC_scored' in fact_lms.columns and 'KC_pct' not in fact_lms.columns:
        fact_lms_copy = fact_lms.copy()
        fact_lms_copy['KC_pct'] = (fact_lms_copy['KC_scored'] / fact_lms_copy['KC_total'].replace(0, 1) * 100).round(1)
        kc_agg = fact_lms_copy.groupby('Intern_ID')['KC_pct'].mean().reset_index()
        lms_agg = lms_agg.drop(columns=['KC_pct'], errors='ignore').merge(kc_agg, on='Intern_ID', how='left')
        
    if 'Test_scored' in fact_lms.columns and 'Test_pct' not in fact_lms.columns:
        fact_lms_copy = fact_lms.copy()
        fact_lms_copy['Test_pct'] = (fact_lms_copy['Test_scored'] / fact_lms_copy['Test_total'].replace(0, 1) * 100).round(1)
        test_agg = fact_lms_copy.groupby('Intern_ID')['Test_pct'].mean().reset_index()
        lms_agg = lms_agg.drop(columns=['Test_pct'], errors='ignore').merge(test_agg, on='Intern_ID', how='left')

    # Aggregate Activity Data
    act_agg = fact_activity.groupby('Intern_ID')['Hours'].sum().reset_index()
    
    # Identify At-Risk internally from heuristics as ground truth for classification
    # We'll use a simplified version: progress < 50 and hours < median hours -> 1 else 0
    median_hours = act_agg['Hours'].median()
    
    merged = lms_agg.merge(act_agg, on='Intern_ID', how='inner')
    merged['Is_At_Risk'] = ((merged['Progress_Numeric'] < 40) | 
                            ((merged['Progress_Numeric'] < 60) & (merged['Hours'] < median_hours * 0.75))).astype(int)
    
    # Fill NA values and ensure KC_pct/Test_pct exist for ML models later
    merged.fillna(0, inplace=True)
    if 'KC_pct' not in merged.columns:
        merged['KC_pct'] = 0.0
    if 'Test_pct' not in merged.columns:
        merged['Test_pct'] = 0.0
        
    merged.columns = [str(c) for c in merged.columns]
    return merged

def get_intern_clusters(fact_lms, fact_activity, n_clusters=3):
    """
    Use K-Means to cluster interns into learning personas based on their performance and hours.
    Returns the dataframe with a 'Cluster' and 'Persona' column, and the silhouette score.
    """
    df = prepare_ml_data(fact_lms, fact_activity)
    if df.empty or len(df) <= n_clusters:
        return pd.DataFrame()
    
    features = ['Progress_Numeric', 'Hours', 'Assg_Ratio', 'KC_pct']
    X = df[features].copy()
    X.columns = [str(c) for c in X.columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Assign personas based on cluster centroids (simplified logic)
    cluster_means = df.groupby('Cluster')[['Progress_Numeric', 'Hours']].mean()
    
    personas = {}
    for c in range(n_clusters):
        prog = cluster_means.loc[c, 'Progress_Numeric']
        hrs = cluster_means.loc[c, 'Hours']
        
        if prog > 75:
            personas[c] = 'High Performers'
        elif prog < 45:
            personas[c] = 'Needs Support'
        elif hrs > cluster_means['Hours'].mean():
            personas[c] = 'Steady Workers'
        else:
            personas[c] = 'Efficient Learners'
            
    df['Persona'] = df['Cluster'].map(personas)
    return df

def predict_test_scores(fact_lms, fact_activity):
    """
    Use Random Forest Regressor to predict 'Test_pct' based on other variables.
    Returns the dataframe with 'Predicted_Test_pct'
    """
    df = prepare_ml_data(fact_lms, fact_activity)
    if df.empty or len(df) < 5:
        return pd.DataFrame()
        
    # We use Test_pct as target
    features = ['Progress_Numeric', 'Hours', 'Assg_Ratio', 'KC_pct']
    target = 'Test_pct'
    
    X = df[features].copy()
    X.columns = [str(c) for c in X.columns]
    y = df[target]
    
    # In a real scenario, we'd split into train/test. Here we'll train on all and predict on all 
    # to show what the model *thinks* they should score, highlighting anomalies.
    rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    df['Predicted_Test_pct'] = rf.predict(X).round(1)
    df['Score_Diff'] = (df['Test_pct'] - df['Predicted_Test_pct']).round(1)
    
    return df

def predict_dropout_risk(fact_lms, fact_activity):
    """
    Use Random Forest Classifier to predict if an intern is At Risk.
    Returns the dataframe with 'Risk_Probability' and 'Predicted_Risk_Class'
    """
    df = prepare_ml_data(fact_lms, fact_activity)
    if df.empty or len(df) < 5:
        return pd.DataFrame()
        
    features = ['Progress_Numeric', 'Hours', 'Assg_Ratio', 'KC_pct', 'Test_pct']
    target = 'Is_At_Risk'
    
    X = df[features].copy()
    X.columns = [str(c) for c in X.columns]
    y = df[target]
    
    if len(y.unique()) < 2:
        # If everyone is same class, model can't really train well
        df['Predicted_Risk_Class'] = y
        df['Risk_Probability'] = 0.5
        return df
        
    rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5, class_weight='balanced')
    rf_clf.fit(X, y)
    
    probs = rf_clf.predict_proba(X)
    # Probability of class 1 (At Risk)
    df['Risk_Probability'] = probs[:, 1].round(3)
    df['Predicted_Risk_Class'] = rf_clf.predict(X)
    
    return df
