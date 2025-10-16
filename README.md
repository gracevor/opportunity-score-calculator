# Opportunity Score Calculator

A Streamlit web application for calculating customer opportunity scores using the Jobs-to-be-Done methodology. This tool helps product teams identify which features or outcomes have the highest potential for improvement based on customer importance and satisfaction ratings.

## 🎯 What It Does

The Opportunity Score Calculator processes customer survey data to identify improvement opportunities using the formula:

**Opportunity = Importance + (Importance - Satisfaction)**

Where:
- **Importance** = (Count of 4s & 5s / Total responses) × 10
- **Satisfaction** = (Count of 4s & 5s / Total responses) × 10

This approach focuses on the "top-2-box" methodology, identifying outcomes that customers rate as very or extremely important/satisfied (4-5 on a 1-5 scale).

## 🚀 Live Demo

[**Try the Calculator**](https://opportunity-score-calculator.streamlit.app/)

## 📊 Features

### Core Functionality
- **Wide Format CSV Support**: Automatically detects importance/satisfaction column pairs
- **1-5 Scale Validation**: Ensures data quality with proper scale validation
- **Smart Missing Data Handling**: Per-outcome analysis maximizes data utilization
- **Interactive Bubble Chart**: Visual representation with GitHub brand colors
- **Comprehensive Reporting**: Detailed calculation breakdowns and data quality metrics

### Data Input Options
- **Survey Data**: Wide format CSV with `imp_` and `sat_` prefixed columns
- **Optional Labels**: Separate CSV for human-readable outcome descriptions
- **Template Downloads**: Built-in templates for proper data formatting

### Export Options
- **Results Table**: CSV download of calculated scores
- **Bubble Chart**: PNG and SVG export options
- **Calculation Details**: Transparent step-by-step breakdowns

## 📁 Data Format Requirements

### Survey Data (Wide Format)
Your CSV must include:
- `respondent_id` column
- Importance columns: `imp_outcome_1`, `imp_outcome_2`, etc.
- Satisfaction columns: `sat_outcome_1`, `sat_outcome_2`, etc.
- All ratings on 1-5 scale (1=Not at all, 5=Extremely)

**Example:**
```csv
respondent_id,imp_outcome_1,imp_outcome_2,sat_outcome_1,sat_outcome_2
R001,5,4,3,2
R002,4,5,4,3
R003,3,4,5,4
```

### Optional Label Mapping
Add human-readable descriptions with a separate CSV:
```csv
outcome_id,label
outcome_1,Reduce time to resolve customer issues
outcome_2,Improve product search functionality
```

## 🔧 How to Use

1. **Upload Data**: Use the sidebar to upload your survey CSV
2. **Add Labels** (Optional): Upload label mapping for better readability
3. **Review Results**: 
   - Check data validation summary
   - Review calculation details
   - Analyze bubble chart patterns
4. **Export**: Download results as CSV or charts as PNG/SVG

## 📈 Interpreting Results

### Opportunity Scores
- **Higher scores** = Greater improvement potential
- **Positive scores** = Importance exceeds satisfaction (improvement gap)
- **Negative scores** = Over-satisfaction (may indicate over-investment)

### Bubble Chart
- **X-axis**: Satisfaction (0-10 scale)
- **Y-axis**: Importance (0-10 scale)  
- **Bubble size**: Opportunity score magnitude
- **Colors**: Different outcomes (uses mapped labels when available)

## 🛠️ Technical Details

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data processing and analysis
- **Altair**: Interactive data visualization
- **Python 3.9+**: Core programming language

### Key Algorithms
- **Auto-detection**: Automatically pairs importance/satisfaction columns
- **Missing data handling**: Per-outcome pairwise deletion
- **Dynamic scaling**: Chart axes adapt to your data range
- **Validation**: Data quality checks

## 📋 Installation & Local Development

```bash
# Clone the repository
git clone https://github.com/gracevor/opportunity-score-calculator.git
cd opportunity-score-calculator

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 References & Further Reading

- [Jobs-to-be-Done Methodology](https://jobs-to-be-done.com/) by Tony Ulwick
- [Top-2-Box Analysis in Customer Research](https://www.questionpro.com/blog/top-2-box-scores/)
- [Customer Satisfaction Measurement Best Practices](https://www.customergauge.com/benchmarks/customer-satisfaction-score)

---

*Built with ❤️ for product teams who want to make data-driven decisions about where to focus their improvement efforts.*
