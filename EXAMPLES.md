# RAG Chatbot - Usage Examples

This document provides comprehensive examples of how to use the RAG Chatbot system, including different types of queries, expected responses, and advanced use cases.

## Table of Contents
- [Document-Based Queries (RAG)](#document-based-queries-rag)
- [Python Code Execution](#python-code-execution)
- [Mixed Queries (RAG + Python)](#mixed-queries-rag--python)
- [Conversation Memory](#conversation-memory)
- [Advanced Use Cases](#advanced-use-cases)
- [API Usage Examples](#api-usage-examples)

## Document-Based Queries (RAG)

### Basic Information Retrieval

**Query:**
```
What are the main topics covered in the uploaded documents?
```

**Expected Response:**
```
Based on the documents in your collection, the main topics covered include:

1. **Machine Learning Fundamentals** - Covering supervised and unsupervised learning algorithms
2. **Neural Networks** - Including deep learning architectures and training methods  
3. **Data Processing** - Techniques for cleaning and preparing datasets
4. **Model Evaluation** - Metrics and validation strategies

*Sources: "ML_Guide.pdf" (pages 1-3, 15-20), "Neural_Networks.pdf" (pages 5-12)*
```

### Specific Information Search

**Query:**
```
How do I implement gradient descent optimization?
```

**Expected Response:**
```
According to the documentation, gradient descent optimization can be implemented using these key steps:

1. **Initialize parameters** randomly or with small values
2. **Forward pass**: Calculate predictions using current parameters
3. **Calculate loss** using your chosen loss function
4. **Backward pass**: Compute gradients of loss with respect to parameters
5. **Update parameters**: θ = θ - α∇θJ(θ) where α is the learning rate

The documents recommend starting with a learning rate between 0.001 and 0.01, and using batch sizes of 32-128 for optimal convergence.

*Source: "Optimization_Methods.pdf" (pages 8-12, 25-27)*
```

### Comparative Analysis

**Query:**
```
Compare the advantages and disadvantages of Random Forest vs Neural Networks
```

**Expected Response:**
```
Based on the uploaded documents, here's a comparison:

**Random Forest Advantages:**
- Less prone to overfitting due to ensemble approach
- Handles missing values automatically
- Provides feature importance rankings
- Fast training and prediction
- Works well with small datasets

**Random Forest Disadvantages:**
- Less effective with very large datasets
- Can struggle with highly correlated features
- Limited expressiveness for complex patterns

**Neural Networks Advantages:**
- Excellent for complex pattern recognition
- Highly flexible architecture
- Superior performance on large datasets
- Can model non-linear relationships effectively

**Neural Networks Disadvantages:**
- Requires large amounts of training data
- Prone to overfitting without regularization
- Computationally expensive
- "Black box" - difficult to interpret

*Sources: "Ensemble_Methods.pdf" (pages 15-20), "Deep_Learning.pdf" (pages 3-8, 45-50)*
```

## Python Code Execution

### Mathematical Calculations

**Query:**
```
python: (25 * 1.15) + (50 * 0.85) - 12.5
```

**Expected Response:**
```
The calculation result is: 58.75

Breaking down the calculation:
- 25 * 1.15 = 28.75
- 50 * 0.85 = 42.5  
- 28.75 + 42.5 - 12.5 = 58.75
```

### Statistical Operations

**Query:**
```
python: import statistics; data = [23, 45, 67, 89, 12, 34, 56, 78, 90, 21]; print(f"Mean: {statistics.mean(data)}, Median: {statistics.median(data)}, StdDev: {round(statistics.stdev(data), 2)}")
```

**Expected Response:**
```
Mean: 51.5, Median: 50.5, StdDev: 28.96

This shows the statistical summary of your dataset [23, 45, 67, 89, 12, 34, 56, 78, 90, 21]:
- **Mean (51.5)**: The average value
- **Median (50.5)**: The middle value when sorted
- **Standard Deviation (28.96)**: Measure of data spread
```

### Natural Language Math

**Query:**
```
Calculate the compound interest if I invest $5000 at 3.5% annual rate for 7 years
```

**Expected Response:**
```
Let me calculate the compound interest for you:

**Calculation:** python: 5000 * (1 + 0.035)**7

**Result:** $6,355.96

**Details:**
- Principal: $5,000
- Annual Interest Rate: 3.5%
- Time Period: 7 years
- **Final Amount:** $6,355.96
- **Interest Earned:** $1,355.96

This assumes annual compounding. The formula used is A = P(1 + r)^t.
```

### Data Analysis

**Query:**
```
python: sales = [120, 150, 180, 200, 165, 190, 220]; growth_rates = [(sales[i] - sales[i-1])/sales[i-1] * 100 for i in range(1, len(sales))]; print(f"Monthly growth rates: {[round(rate, 1) for rate in growth_rates]}%")
```

**Expected Response:**
```
Monthly growth rates: [25.0, 20.0, 11.1, -17.5, 15.2, 15.8]%

**Analysis of your sales data:**
- **Strongest growth:** 25.0% (Month 1 to 2)  
- **Decline period:** -17.5% (Month 4 to 5)
- **Recovery trend:** 15.2% and 15.8% in final months
- **Average growth:** ~11.6% per month
```

## Mixed Queries (RAG + Python)

### Document Analysis with Calculations

**Query:**
```
According to the financial report, what was the ROI if the initial investment was $100,000? Also calculate what the ROI would be with a 15% increase in returns.
```

**Expected Response:**
```
Based on the financial report, the company achieved a net return of $125,000 on the initial investment.

**Current ROI Calculation:**
python: roi_current = (125000 - 100000) / 100000 * 100

**Current ROI:** 25.0%

**With 15% increase in returns:**
python: new_return = 125000 * 1.15; roi_increased = (new_return - 100000) / 100000 * 100

**Improved ROI:** 43.75%

**Summary:**
- Current return: $125,000 (25.0% ROI)
- With 15% increase: $143,750 (43.75% ROI)  
- **Additional profit:** $18,750

*Source: "Q3_Financial_Report.pdf" (page 12)*
```

### Research Data Validation

**Query:**
```
The research paper mentions a 95% confidence interval. Can you verify the statistical significance if the sample size was 200 and the margin of error was 3.5%?
```

**Expected Response:**
```
According to the research paper, the study used a 95% confidence interval methodology.

**Statistical Verification:**
python: import math; z_score = 1.96; margin_error = 0.035; n = 200; std_error = margin_error / z_score; print(f"Standard error: {round(std_error, 4)}, Required sample size check: {round((z_score/margin_error)**2 * 0.25)}")

**Verification Results:**
- **Standard Error:** 0.0178
- **Minimum Required Sample Size:** 196 (for 95% CI with 3.5% margin)
- **Actual Sample Size:** 200 ✅

**Conclusion:** The study's sample size of 200 is adequate for achieving a 95% confidence interval with 3.5% margin of error. The results are statistically valid.

*Source: "Research_Methodology.pdf" (pages 15-17)*
```

## Conversation Memory

### Context Preservation

**Conversation Example:**

**User:** "What is machine learning?"

**Assistant:** "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task..."

**User:** "What are its main types?"

**Assistant:** "Based on our previous discussion about machine learning, there are three main types:

1. **Supervised Learning** - Uses labeled training data to learn mapping functions
2. **Unsupervised Learning** - Finds patterns in unlabeled data  
3. **Reinforcement Learning** - Learns through interaction and feedback

Each type serves different purposes and has specific applications..."

**User:** "Calculate the accuracy if I have 850 correct predictions out of 1000 total"

**Assistant:** "For your machine learning model performance:

python: accuracy = 850 / 1000 * 100

**Accuracy:** 85.0%

This is quite good! In the context of our machine learning discussion:
- 85% accuracy indicates strong model performance
- This exceeds typical benchmarks for many supervised learning tasks
- Consider also evaluating precision, recall, and F1-score for comprehensive assessment"

### Multi-turn Technical Discussion

**Turn 1:**
**User:** "Explain neural network backpropagation"

**Turn 2:** 
**User:** "How do I calculate the gradient for the hidden layer?"

**Turn 3:**
**User:** "If I have learning rate 0.01 and gradient -2.5, what's the parameter update?"

**Assistant:** "Following our backpropagation discussion, for your parameter update:

python: learning_rate = 0.01; gradient = -2.5; parameter_update = -learning_rate * gradient

**Parameter Update:** +0.025

**Explanation:**
- With gradient of -2.5, the parameter should move in the opposite direction
- Update = -(0.01) × (-2.5) = +0.025
- This means the parameter will increase by 0.025

This positive update makes sense because the negative gradient indicates the loss function is decreasing in the negative direction, so we move the parameter in the positive direction to minimize loss."

## Advanced Use Cases

### Financial Analysis Workflow

**Query:**
```
Analyze the investment portfolio data from the uploaded spreadsheet and calculate the Sharpe ratio for each asset. Also recommend portfolio allocation based on risk-adjusted returns.
```

**Expected Response:**
```
Based on the portfolio data in your uploaded documents, I've analyzed the investment performance:

**Individual Asset Analysis:**
- **Stock A**: Return 12.5%, Volatility 18%, Sharpe Ratio: 0.69
- **Stock B**: Return 8.3%, Volatility 12%, Sharpe Ratio: 0.69  
- **Bond Fund**: Return 4.1%, Volatility 5%, Sharpe Ratio: 0.82

**Calculations:**
python: sharpe_a = (12.5 - 2.0) / 18; sharpe_b = (8.3 - 2.0) / 12; sharpe_bonds = (4.1 - 2.0) / 5
python: print(f"Sharpe Ratios - Stock A: {round(sharpe_a, 2)}, Stock B: {round(sharpe_b, 2)}, Bonds: {round(sharpe_bonds, 2)}")

**Recommended Allocation:**
Based on Modern Portfolio Theory and risk-adjusted returns:
- **Bond Fund**: 40% (highest Sharpe ratio, stability)
- **Stock B**: 35% (good risk-adjusted return, lower volatility)
- **Stock A**: 25% (growth potential, higher risk)

*Source: "Portfolio_Analysis_2024.pdf" (pages 5-12)*
```

### Research Data Processing

**Query:**
```
Extract the experimental results from the research paper and perform a t-test to validate the claimed statistical significance.
```

**Expected Response:**
```
From the research paper, I found the following experimental data:

**Experimental Results:**
- **Control Group**: Mean = 72.3, SD = 8.5, n = 45
- **Treatment Group**: Mean = 78.9, SD = 9.2, n = 42

**Statistical Analysis:**
python: import math; m1, m2 = 72.3, 78.9; s1, s2 = 8.5, 9.2; n1, n2 = 45, 42; pooled_se = math.sqrt((s1**2/n1) + (s2**2/n2)); t_stat = (m2 - m1) / pooled_se; print(f"Difference: {m2-m1}, SE: {round(pooled_se, 3)}, t-statistic: {round(t_stat, 3)}")

**T-test Results:**
- **Mean Difference**: 6.6 points
- **Standard Error**: 1.864  
- **T-statistic**: 3.540
- **Degrees of Freedom**: ~85
- **P-value**: < 0.001 (highly significant)

**Conclusion:** The results confirm statistical significance (p < 0.001), supporting the paper's claims. The treatment shows a meaningful improvement of 6.6 points with high confidence.

*Source: "Clinical_Trial_Results.pdf" (Table 3, page 18)*
```

## API Usage Examples

### Health Check Integration

```bash
# Check system health before automated queries
curl -f http://localhost:8080/health

# Expected Response:
{
  "status": "healthy",
  "timestamp": "2024-08-27T12:00:00Z",
  "services": {
    "qdrant": "healthy",
    "ollama": "healthy", 
    "pyexec": "healthy"
  }
}
```

### Direct Python Execution

```bash
# Execute Python via API
curl -X POST http://localhost:8001/eval \
  -H "Content-Type: application/json" \
  -d '{"expression": "sum([1, 2, 3, 4, 5]) / len([1, 2, 3, 4, 5])"}'

# Expected Response:
{
  "result": 3.0,
  "expression": "sum([1, 2, 3, 4, 5]) / len([1, 2, 3, 4, 5])",
  "execution_time": 0.002,
  "status": "success"
}
```

### Batch Document Processing

```bash
# Add documents and trigger ingestion
cp new_research_paper.pdf docs/
docker compose run --rm ingest

# Expected Log Output:
Processing documents...
✓ new_research_paper.pdf: 45 pages, 234 chunks created
✓ Embeddings generated: 234 vectors
✓ Stored in Qdrant collection: corpus_pdf
Ingestion completed in 3.2 seconds
```

## Error Handling Examples

### Invalid Python Code

**Query:**
```
python: import os; os.system("rm -rf /")
```

**Expected Response:**
```
❌ **Security Error**: The Python expression contains restricted operations.

**Details:**
- Attempted to use: `import` statement (blocked)
- Attempted to use: `os.system()` (dangerous function)
- **Reason**: File system operations are disabled for security

**Allowed operations:**
- Mathematical calculations
- Statistical functions
- Data manipulation
- String operations
- Basic control structures

**Try instead:** `python: 2 + 2` or `python: sum([1, 2, 3, 4, 5])`
```

### Timeout Handling

**Query:**
```
python: while True: pass
```

**Expected Response:**
```
⏱️ **Timeout Error**: Python execution exceeded the time limit.

**Details:**
- **Time limit**: 5 seconds
- **Reason**: Infinite loop detected
- **Status**: Execution terminated safely

**Tip**: Avoid infinite loops and recursive functions. For large calculations, consider breaking them into smaller parts.
```

### No Relevant Documents

**Query:**
```
What is the weather forecast for tomorrow?
```

**Expected Response:**
```
🔍 **No Relevant Information Found**

I couldn't find information about weather forecasts in your uploaded documents. The current document collection appears to focus on:

- Machine learning and AI topics
- Financial analysis and reports  
- Research methodologies

**Suggestions:**
- Upload weather-related documents for analysis
- Try asking about topics covered in your documents
- Use Python for weather-related calculations: `python: temperature_celsius = (75 - 32) * 5/9`
```

## Performance Benchmarks

### Response Time Examples

| Query Type | Complexity | Avg Response Time | Example |
|------------|------------|-------------------|---------|
| Simple RAG | Low | 2-4 seconds | "What is machine learning?" |
| Complex RAG | High | 5-8 seconds | Multi-document comparative analysis |
| Python Math | Low | 0.5-1 second | `python: 2 + 2` |
| Python Complex | Medium | 1-3 seconds | Statistical calculations with data |
| Mixed Query | High | 6-10 seconds | RAG + Python analysis |

### Resource Usage

- **Memory**: 2-4 GB typical usage
- **CPU**: Peaks during LLM inference
- **Disk**: Grows with document collection size
- **Network**: Minimal after initial model download

---

## Testing Your Setup

Use these examples to verify your installation:

1. **Basic RAG**: Ask about your uploaded documents
2. **Math**: Try `python: 10 * 5 + 3`  
3. **Statistics**: `python: import statistics; statistics.mean([1,2,3,4,5])`
4. **Memory**: Ask follow-up questions referencing previous responses
5. **Mixed**: Combine document queries with calculations

For more examples and advanced use cases, see the test files in `tests/fixtures/qa_dataset.json`.

---

**Last Updated**: August 2024  
**Compatible with**: RAG Chatbot v1.0.0