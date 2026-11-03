
import pandas as pd

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/H2.csv")

# ==========================================
# 2. Define Target
# ==========================================

TARGET = "IsCanceled"

# ==========================================
# 3. Define Selected Features
# ==========================================

FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
    "ADR",
]

# ==========================================
# 4. Prepare X and y
# ==========================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ==========================================
# 5. Handle Missing Values
# ==========================================

X["Children"] = X["Children"].fillna(0)
X["Country"] = X["Country"].fillna("Unknown")

# ==========================================
# 6. Verify Data
# ==========================================

print("=" * 50)
print("FEATURE SELECTION COMPLETED")
print("=" * 50)

print(f"\nDataset Shape: {df.shape}")
print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

print("\nSelected Features:")
for feature in FEATURES:
    print(f"- {feature}")

print("\nTarget Distribution:")
print(y.value_counts())

print("\nMissing Values After Cleaning:")
print(X.isnull().sum().sum())

print("\nFeature Selection Completed Successfully!")
