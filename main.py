
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("social_media_data.csv")

# 2. Create a total engagement column
df["total_engagement"] = (
    df["likes"] + df["comments"] + df["shares"]
)

# Calculate engagement rate as a percentage of reach
df["engagement_rate"] = (
    df["total_engagement"] / df["reach"]
) * 100

# 3. Display basic analysis
print("\n===== SOCIAL MEDIA ANALYTICS =====")

print("\nTotal number of posts:", len(df))

print("\nAverage likes:", round(df["likes"].mean(), 2))
print("Average comments:", round(df["comments"].mean(), 2))
print("Average shares:", round(df["shares"].mean(), 2))

print("\nAverage engagement by post type:")
print(
    df.groupby("post_type")["total_engagement"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print("\nAverage engagement rate by post type:")
print(
    df.groupby("post_type")["engagement_rate"]
    .mean()
    .round(2)
)

print("\nBest-performing post in this sample:")
best_post = df.loc[df["total_engagement"].idxmax()]
print(best_post[["post_type", "posting_hour",
                 "total_engagement", "engagement_rate"]])

# 4. Set chart style
sns.set_theme(style="whitegrid")

# Chart 1: Average engagement by post type
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="post_type",
    y="total_engagement",
    hue="post_type",
    legend=False,
    errorbar=None
)
plt.title("Average Engagement by Post Type")
plt.xlabel("Post Type")
plt.ylabel("Average Likes + Comments + Shares")
plt.tight_layout()
plt.savefig("engagement_by_post_type.png")
plt.show()

# Chart 2: Engagement by posting hour
hourly_data = df.groupby("posting_hour")[
    "total_engagement"
].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=hourly_data,
    x="posting_hour",
    y="total_engagement",
    marker="o"
)
plt.title("Average Engagement by Posting Hour")
plt.xlabel("Posting Hour (24-hour format)")
plt.ylabel("Average Total Engagement")
plt.xticks(range(9, 22))
plt.tight_layout()
plt.savefig("engagement_by_hour.png")
plt.show()

# Chart 3: Relationship between reach and engagement
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="reach",
    y="total_engagement",
    hue="post_type",
    s=100
)
plt.title("Reach vs Total Engagement")
plt.xlabel("Reach")
plt.ylabel("Total Engagement")
plt.tight_layout()
plt.savefig("reach_vs_engagement.png")
plt.show()

print("\nAnalysis completed successfully!")
print("Three charts have been saved in your project folder.")

