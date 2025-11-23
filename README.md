# Feature importance by mean(|SHAP|)
mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
display(shap_importance.head(30))

# Dependence plot for top feature
top_feat = shap_importance.index[0]
shap.dependence_plot(top_feat, shap_vals_pos, X_test_p, feature_names=feature_names, show=True)

# 7) Local explanations: pick specific instances to analyze
# We'll choose:
#  - a correctly predicted default (TP)
#  - a false negative (model predicted low risk but actually defaulted)
#  - a high-risk non-default (FP)
pred_labels = (y_pred_proba >= 0.5).astype(int)
test_idx = X_test.index  # original indices
results_df = pd.DataFrame({
    'index': test_idx,
    'y_true': y_test.values,
    'y_proba': y_pred_proba,
    'y_pred': pred_labels
}).set_index('index')

# select one example of each interesting case if exists
cases = {}
for name, cond in [
    ("true_positive", (results_df.y_true==1) & (results_df.y_pred==1)),
    ("false_negative",(results_df.y_true==1) & (results_df.y_pred==0)),
    ("false_positive",(results_df.y_true==0) & (results_df.y_pred==1))
]:
    sel = results_df[cond]
    if not sel.empty:
        cases[name] = sel.index[0]

cases
# Show the chosen rows
for k, idx in cases.items():
    print(k, idx, results_df.loc[idx].to_dict())

# Get SHAP explanation for a chosen instance (example: false_negative)
chosen_idx = list(cases.values())[0]  # pick first available
local_shap = shap_vals_pos[X_test.index.get_loc(chosen_idx)]
# Force plot for local explanation
shap.force_plot(explainer.expected_value, local_shap, X_test_p[X_test.index.get_loc(chosen_idx)], feature_names=feature_names, matplotlib=True)

# 8) LIME local explanation for same instance (model-agnostic)
# Need a prediction function that accepts raw preprocessed arrays
def predict_proba_for_lime(X_array):
    # X_array already preprocessed to match training preproc output
    return np.vstack([1 - model.predict(X_array), model.predict(X_array)]).T

# But LimeTabularExplainer expects raw X_train in original form; easier: wrap with preproc inside predict_fn
# Create a wrapper that accepts raw rows (shape: (n_samples, n_features_original))
def model_predict_proba_raw(X_raw):
    Xp = preproc.transform(pd.DataFrame(X_raw, columns=X.columns))
    probs = model.predict(Xp)
    return np.vstack([1-probs, probs]).T

# Build Lime explainer using training raw data (converted to numpy)
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=list(X.columns),
    class_names=['no_default','default'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# pick instance in original raw form
inst_pos = X_test.loc[chosen_idx].values
lime_exp = lime_explainer.explain_instance(inst_pos, model_predict_proba_raw, num_features=10)
print("LIME explanation (top features):")
print(lime_exp.as_list())

# 9) Quantitative comparison SHAP vs LIME for that instance
# Get LIME attributions in a feature vector aligned to feature_names
lime_list = lime_exp.as_list()
# map LIME names to original columns — careful: LIME uses thresholds for continuous/discrete and bin labels
# We'll build a best-effort mapping: for each original feature, sum absolute contribution if any LIME entry mentions it.
lime_attrib = pd.Series(0.0, index=feature_names)
for feat_label, weight in lime_list:
    # find which original feature it refers to: take substring before any operator or ' ' or '='
    orig_feat = feat_label.split(' ')[0].split('=')[0].split('<=')[0].split('>')[0]
    # Try to match to any feature_name that startswith orig_feat
    matches = [f for f in feature_names if f.startswith(orig_feat)]
    if matches:
        # If multiple OHE features matched, assign to all equally or to first — we'll assign to first match for simplicity
        lime_attrib[matches[0]] += weight
    else:
        # fallback: ignore if no match
        pass


# Normalize and compare vectors
shap_vec = pd.Series(local_shap, index=feature_names)
# Align (ensure same ordering)
lime_vec = lime_attrib
# compute rank correlation on absolute values (importance)
mask = (shap_vec.abs()>0) | (lime_vec.abs()>0)
if mask.sum() > 1:
    spearman_corr, _ = spearmanr(shap_vec.abs()[mask], lime_vec.abs()[mask])
    pearson_corr, _ = pearsonr(shap_vec.abs()[mask], lime_vec.abs()[mask])
else:
    spearman_corr, pearson_corr = np.nan, np.nan

print("Spearman corr (abs attributions):", spearman_corr)
print("Pearson corr (abs attributions):", pearson_corr)

# Also check sign agreement on top-k features
topk = 10
shap_topk = shap_vec.abs().sort_values(ascending=False).head(topk).index
agree = 0
count = 0
for f in shap_topk:
    if lime_vec[f] != 0:
        count += 1
        if np.sign(shap_vec[f]) == np.sign(lime_vec[f]):
            agree += 1
sign_agreement = agree / count if count>0 else np.nan
print(f"Sign agreement among top-{topk} features (where both available): {sign_agreement} (based on {count} comparable features)")

# 10) Batch quantitative comparison for N random instances
def compare_shap_lime_for_index(idx):
    # get SHAP vector
    loc = X_test.index.get_loc(idx)
    s = pd.Series(shap_vals_pos[loc], index=feature_names)
    # LIME
    raw_inst = X_test.loc[idx].values
    lime_e = lime_explainer.explain_instance(raw_inst, model_predict_proba_raw, num_features=30)
    lime_s = pd.Series(0.0, index=feature_names)
    for feat_label, weight in lime_e.as_list():
        orig_feat = feat_label.split(' ')[0].split('=')[0].split('<=')[0].split('>')[0]
        matches = [f for f in feature_names if f.startswith(orig_feat)]
        if matches:
            lime_s[matches[0]] += weight
    # compute correlations
    mask = (s.abs()>0) | (lime_s.abs()>0)
    if mask.sum() > 1:
        sp = spearmanr(s.abs()[mask], lime_s.abs()[mask]).correlation
    else:
        sp = np.nan
    return sp

N = 30
sample_idxs = np.random.choice(X_test.index, size=min(N, len(X_test)), replace=False)
corrs = []
for idx in sample_idxs:
    corrs.append(compare_shap_lime_for_index(idx))
pd.Series(corrs).describe()

# 11) Business translation
# Summarize top global drivers from shap_importance, e.g.
print("Top global drivers (SHAP):")
print(shap_importance.head(15))

# Produce cohort analyses: e.g., average SHAP by binned income / age / loan amount
# Example for numeric column 'age' -- adapt to your column names
if 'age' in X.columns:
    # create bins on raw X_test['age']
    bins = pd.qcut(X_test['age'], q=5, duplicates='drop')
    avg_shap_by_bin = []
    for b in bins.unique().categories:
        idxs = bins==b
        avg_shap_by_bin.append({
            'bin': str(b),
            'avg_shap_top_feature': shap_vals_pos[idxs.values][:, feature_names.index(top_feat)].mean()
        })
    display(pd.DataFrame(avg_shap_by_bin))

# End of notebook
