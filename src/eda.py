
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Setup
# ==========================================

df = pd.read_csv("data/H2.csv")

output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("STARTING EDA VISUALIZATIONS")
print("=" * 60)


# ==========================================
# 2. Cancellation Distribution
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="IsCanceled"
)

plt.title("Booking Cancellation Distribution")
plt.xlabel("Is Canceled (0 = No, 1 = Yes)")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.savefig(f"{output_dir}/cancellation_distribution.png")
plt.show()
plt.close()


# ==========================================
# 3. Lead Time Distribution
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="LeadTime",
    bins=40,
    kde=True
)

plt.title("Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.savefig(f"{output_dir}/lead_time_distribution.png")
plt.show()
plt.close()


# ==========================================
# 4. ADR Distribution
# ==========================================

# Exclude ADR values above 500 from this plot only
# Original dataset remains unchanged

adr_for_plot = df[df["ADR"] <= 500]

plt.figure(figsize=(9, 5))

sns.histplot(
    data=adr_for_plot,
    x="ADR",
    bins=40,
    kde=True
)

plt.title("Average Daily Rate (ADR) Distribution")
plt.xlabel("ADR")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.savefig(f"{output_dir}/adr_distribution.png")
plt.show()
plt.close()

print(
    "ADR values excluded from visualization only:",
    len(df) - len(adr_for_plot)
)


# ==========================================
# 5. Bookings Per Month
# ==========================================

month_order = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
]

monthly_bookings = (
    df["ArrivalDateMonth"]
    .value_counts()
    .reindex(month_order)
)

plt.figure(figsize=(11, 5))

sns.barplot(
    x=monthly_bookings.index,
    y=monthly_bookings.values
)

plt.title("Number of Bookings Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Bookings")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/bookings_per_month.png")
plt.show()
plt.close()


# ==========================================
# 6. Lead Time vs Cancellation
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="IsCanceled",
    y="LeadTime",
    showfliers=False
)

plt.title("Lead Time vs Booking Cancellation")
plt.xlabel("Is Canceled (0 = No, 1 = Yes)")
plt.ylabel("Lead Time (Days)")

plt.tight_layout()
plt.savefig(f"{output_dir}/lead_time_vs_cancellation.png")
plt.show()
plt.close()


# ==========================================
# 7. Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(15, 11))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.show()
plt.close()


# ==========================================
# 8. Finish
# ==========================================

print("\n" + "=" * 60)
print("EDA VISUALIZATIONS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved Figures:")

for filename in os.listdir(output_dir):
    print(filename)
