
# Social Media Analytics

## Project Overview
Social Media Analytics is a Python-based project that analyzes social media post engagement and uses machine learning to predict total engagement.

The project explores how post type, posting time, and reach relate to engagement.

## Objectives
- Analyze social media engagement metrics.
- Compare engagement across different post types.
- Explore engagement patterns by posting hour.
- Build a machine learning model to predict total engagement.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Random Forest Regression

## Dataset
The dataset contains post type, posting hour, reach, likes, comments, and shares.

**Note:** The included dataset is illustrative sample data. Results and predictions are for educational purposes and should not be treated as real-world performance estimates.

## Project Features
- Calculates total engagement and engagement rate.
- Compares average engagement by post type.
- Visualizes engagement by posting hour.
- Plots reach against engagement.
- Trains a machine learning model to estimate total engagement.

## Project Files
- `main.py` — Performs data analysis and generates charts.
- `predict_engagement.py` — Trains and tests the prediction model.
- `social_media_data.csv` — Sample dataset.
- `engagement_by_post_type.png` — Engagement comparison by post type.
- `engagement_by_hour.png` — Engagement by posting hour.
- `reach_vs_engagement.png` — Reach versus engagement visualization.

## How to Run

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Run the analysis:

```bash
python main.py
```

Run the prediction model:

```bash
python predict_engagement.py
```

## Future Improvements
- Use a larger dataset collected from reliable sources.
- Evaluate the model with more data and suitable metrics.
- Explore additional features and improve prediction performance.

## Author
Shakshi270906
