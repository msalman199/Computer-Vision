import pandas as pd
import matplotlib.pyplot as plt

# 1. Load packet data and remove rows without IP addresses
df = pd.read_csv("traffic.csv")
df = df.dropna(subset=["ip.src", "ip.dst"])

# 2. Count packets per source IP (Top 10)
top_sources = df["ip.src"].value_counts().head(10)

# 3. Check for suspicious activity (e.g., potential flooding > 100 packets)
suspicious = top_sources[top_sources > 100]
if not suspicious.empty:
    print("\n⚠️  Potential Threats Detected:")
    print(suspicious)
else:
    print("\n✅  No suspicious IPs found.")

# 4. Generate and save bar chart
plt.figure(figsize=(10, 5))
top_sources.plot(kind="bar", color="skyblue")
plt.title("Top 10 Source IPs by Packet Count")
plt.xlabel("Source IP")
plt.ylabel("Packets")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("traffic_bar_chart.png")
print("\n📊 Chart saved as 'traffic_bar_chart.png'")

plt.show()
