# Real-Time SWOT Analysis Project  
## Simple One-Page Explanation

---

### What We Built

We built a program that uses **machine learning** to automatically classify a business situation as a **Strength**, **Weakness**, **Opportunity**, or **Threat** (SWOT). Instead of manually analyzing data, the program learns from past examples and makes predictions for new situations.

---

### Why It Matters

Traditional SWOT analysis is done by hand and can be slow or subjective. Our program takes business metrics (like marketing performance, finances, customer feedback, etc.), feeds them into a trained model, and instantly suggests which SWOT category best fits the situation. This helps analysts work faster and more consistently.

---

### What Goes In, What Comes Out

**Inputs (10 metrics):** Marketing condition, financial performance, customer feedback, industry competition, product quality, consumer behavior, expansion ability, uncontrollable factors, market saturation, and marketing strategies. Each is a numeric score (e.g., 0–1).

**Output:** A single label **S**, **W**, **O**, or **T** indicating how the program classifies that situation.

---

### How It Works (Simple Version)

1. **We created a dataset**: (`swot.arff`) contains many past examples with 10 metrics and their correct SWOT labels.

2. **The program learns from the data** : About two-thirds of the data is used to “train” a decision tree. The tree learns rules like: “When marketing is high and finances are low, it’s usually a Weakness.”

3. **We test how good it is** : The remaining third of the data is used to see how often the program guesses correctly. This tells us how reliable it is.

4. **We can make new predictions** : After training, the program can take a new set of 10 metric values and predict whether that situation is S, W, O, or T.

---

### What We Used

- **Java** — The programming language.
- **Weka** — A free, open-source machine learning toolkit. We use its J48 decision tree algorithm to learn the rules from data.
- **J48 Decision Tree** — A method that builds a tree of simple rules (e.g., “if X > 0.5 then …”). It’s easy to understand and works well for this kind of classification.

---

### In Plain Terms

Think of it like teaching someone to recognize fruit. You show them many examples: “This is an apple, this is an orange.” After enough examples, they can look at a new fruit and guess what it is. Our program does the same thing—but with business metrics instead of fruit, and with S/W/O/T instead of fruit names. We use a decision tree because it learns clear, interpretable rules, which is helpful when explaining results to business users.

---

### Authors & Updates

*Created by Aleeza Bhatti & Yousra Esseddiqi (July 2024), Guided by Taesik Kim*
