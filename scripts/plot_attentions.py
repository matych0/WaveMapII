def plot_attention(inference_df, bins=30):
    if isinstance(inference_df, pd.DataFrame):
        attentions = np.concatenate(inference_df["attentions"].values, axis=0)
    elif isinstance(inference_df, pd.Series):
        attentions = inference_df["attentions"]
    g = sns.histplot(data=attentions, bins=bins)
    g.set_yscale("log")
    plt.title("Attention Weights Distribution")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.show()


def plot_low_high_risk(inference_df, n_patients=10):
    # Step 1: Compute unique patient-level risks
    patient_risks = (
        inference_df.groupby("patient_path")["risk"]
        .mean()
        .reset_index()
    )

    # Step 2: Select patient paths with lowest and highest risks
    low_risk_patients = patient_risks.nsmallest(n_patients, "risk")["patient_path"].values
    high_risk_patients = patient_risks.nlargest(n_patients, "risk")["patient_path"].values

    # Step 3: Filter the inference_df for those patients
    low_risk_df = inference_df[inference_df["patient_path"].isin(low_risk_patients)]
    high_risk_df = inference_df[inference_df["patient_path"].isin(high_risk_patients)]

    # Step 4: Extract and concatenate attention weights
    low_risk_attentions = low_risk_df["attention"].values
    high_risk_attentions = high_risk_df["attention"].values

    # Step 5: Prepare DataFrame for Seaborn
    attentions_df = pd.DataFrame({
        "attention": np.concatenate([low_risk_attentions, high_risk_attentions]),
        "class": ["low risk"] * len(low_risk_attentions) + ["high risk"] * len(high_risk_attentions)
    })

    # Step 6: Plot
    g = sns.histplot(data=attentions_df, x="attention", hue="class", bins=30, multiple="dodge", palette="vlag")
    g.set_yscale("log")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.title(f"Attention Weights for {n_patients} Lowest and Highest Risk Patients")
    plt.show()