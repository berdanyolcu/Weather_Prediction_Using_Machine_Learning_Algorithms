<h2>Methodology</h2>

<ol>
  <li>
    <strong>Data Preprocessing:</strong>
    <ul>
      <li>Converted categorical features (<code>RainToday</code>, <code>RainTomorrow</code>, <code>WindGustDir</code>, <code>WindDir9am</code>, <code>WindDir3pm</code>) into numerical values using one-hot encoding.</li>
      <li>Replaced <code>Yes/No</code> in <code>RainToday</code> and <code>RainTomorrow</code> with binary values (<code>1</code> for Yes, <code>0</code> for No).</li>
      <li>Dropped irrelevant columns (e.g., <code>Date</code>).</li>
      <li>Normalized the data where required for algorithms like SVM.</li>
    </ul>
  </li>
  <li>
    <strong>Train-Test Split:</strong>
    <p>Divided the data into training and testing sets using an 80-20 split to evaluate model performance on unseen data.</p>
  </li>
  <li>
    <strong>Machine Learning Models:</strong>
    <ul>
      <li><strong>K-Nearest Neighbors (KNN):</strong>
        <ul>
          <li>Tuned <code>n_neighbors</code> for better accuracy.</li>
          <li>Evaluated accuracy, Jaccard Index, and F1 Score.</li>
        </ul>
      </li>
      <li><strong>Decision Tree:</strong>
        <p>Built a decision tree classifier and evaluated its metrics.</p>
      </li>
      <li><strong>Linear Regression:</strong>
        <ul>
          <li>Predicted the target variable using regression.</li>
          <li>Computed metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.</li>
        </ul>
      </li>
      <li><strong>Logistic Regression:</strong>
        <ul>
          <li>Performed binary classification.</li>
          <li>Computed accuracy, Jaccard Index, F1 Score, and Log Loss.</li>
        </ul>
      </li>
      <li><strong>Support Vector Machine (SVM):</strong>
        <p>Used a linear kernel and evaluated the model's performance.</p>
      </li>
    </ul>
  </li>
  <li>
    <strong>Evaluation Metrics:</strong>
    <ul>
      <li><strong>Classification Metrics:</strong> Accuracy, Jaccard Index, F1 Score, and Log Loss.</li>
      <li><strong>Regression Metrics:</strong> MAE, MSE, RMSE, and R² Score.</li>
    </ul>
  </li>
</ol>

<h2>Conclusion</h2>

<p>This project demonstrates the application of multiple machine learning techniques to predict weather conditions, specifically whether it will rain tomorrow (<code>RainTomorrow</code>). Each model has shown its strengths and weaknesses:</p>

<ul>
  <li><strong>Logistic Regression</strong> and <strong>SVM</strong> provided the most reliable results for classification tasks, achieving high accuracy and balanced performance across metrics.</li>
  <li><strong>Linear Regression</strong> was helpful for understanding the relationships between variables but not ideal for binary classification tasks.</li>
  <li>The <strong>KNN model</strong> achieved good accuracy but struggled with correctly classifying the positive class, as indicated by lower Jaccard Index and F1 scores.</li>
  <li><strong>Decision Tree</strong> provided flexibility in handling the dataset but had performance variability based on hyperparameter settings.</li>
</ul>

<p>Overall, the results highlight the importance of selecting and tuning machine learning algorithms based on the specific characteristics of the dataset and the task at hand.</p>

<h3>Future Work</h3>
<ul>
  <li>Perform hyperparameter tuning for models like Decision Tree, SVM, and KNN to optimize performance further.</li>
  <li>Incorporate additional features, such as seasonal trends or geographical data, to enhance prediction accuracy.</li>
  <li>Experiment with ensemble methods like Random Forest and Gradient Boosting to combine the strengths of individual models.</li>
</ul>
