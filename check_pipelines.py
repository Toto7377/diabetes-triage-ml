from ml.data import load_data, split
from ml.pipeline_v01 import make_v01
from ml.pipeline_v02 import make_v02
from sklearn.metrics import mean_squared_error

print(" Loading data...")
X, y = load_data()
Xtr, Xte, ytr, yte = split(X, y)

# --- v0.1: 
print("\n Testing v0.1 pipeline:")
pipe1 = make_v01()
pipe1.fit(Xtr, ytr)
pred1 = pipe1.predict(Xte)
rmse1 = mean_squared_error(yte, pred1, squared=False)
print(f" v0.1 RMSE: {rmse1:.3f}")

# --- v0.2: 
print("\n Testing v0.2 pipeline:")
pipe2 = make_v02()
pipe2.fit(Xtr, ytr)
pred2 = pipe2.predict(Xte)
rmse2 = mean_squared_error(yte, pred2, squared=False)
print(f" v0.2 RMSE: {rmse2:.3f}")

# --- Comparison ---
delta = rmse1 - rmse2
if delta > 0:
    print(f"\n Improvement: {delta:.3f} lower RMSE (better model)")
else:
    print(f"\n No improvement (Δ={delta:.3f}) — expected small difference for RidgeCV.")
