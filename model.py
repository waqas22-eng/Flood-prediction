import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("data.csv")

# Split input and output
X = df[["rainfall", "river_level", "elevation", "distance", "history"]]
y = df["flood"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
# GRAPH SECTION
st.subheader("📊 Data Visualization")

labels = ['Rainfall', 'River Level', 'Elevation', 'Distance']
values = [rainfall, river_level, elevation, distance]

fig, ax = plt.subplots()
ax.bar(labels, values)

st.pyplot(fig)
# Probability Pie Chart
fig2, ax2 = plt.subplots()

ax2.pie([probability, 1-probability],
        labels=["Flood Risk", "Safe"],
        autopct='%1.1f%%')

st.pyplot(fig2)