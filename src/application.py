from flask import Flask, request, jsonify , json , send_file, send_from_directory
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import io
from sklearn.preprocessing import LabelEncoder
import matplotlib
# Set the backend to 'Agg' which doesn't require a GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import json
import os

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Add this route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# If you want to restrict CORS to specific origins, you can do this:
# CORS(app, origins="http://localhost:3000")

global df

# df = pd.DataFrame({
#     'Column1': [1, 2, 100000],
#     'Column2': [1, 5, 6321],
#     'Column3': [1, 8, 9321]
# })


@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        file_path = 'uploads/' + file.filename
        file.save(file_path)
        df = pd.read_csv(file_path)
        print("The file is successfully uploaded .")
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200
    
    return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400


@app.route('/removeduplicates', methods=['POST'])
def removeDuplicates():
    global df  # Ensure 'df' is globally accessible
    if df is not None:
        duplicate_count = int(df.duplicated().sum())
        df.drop_duplicates(inplace=True)
        return jsonify({'duplicate_count': duplicate_count}), 200
    else:
        return jsonify({'error': 'No data available to process.'}), 400


@app.route('/analyze', methods=['POST'])
def analyze_file():
    global df
    if df is None:
        return jsonify(["No data available. Please upload a CSV file first."]), 400
    
    # Get column names from the dataframe
    columns = df.columns.tolist()
    print("Sending columns:", columns)
    return jsonify(columns)


@app.route('/impute', methods=['POST'])
def replaceMissing():
    global df
    strategy = request.form.get('strategy')
    selected_columns = request.form.get('columns')  # JSON string from the frontend
    selected_columns = json.loads(selected_columns)  # Deserialize JSON string to a Python list
    
    # Use SimpleImputer with NaN as the missing value
    imputer = SimpleImputer(strategy=strategy, missing_values=np.nan)            
    for column in selected_columns:
        # First, convert the column to numeric if possible
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            pass  # Keep as is if can't convert to numeric
            
        # Apply imputation
        df[column] = imputer.fit_transform(df[[column]]).ravel()
    
    print(df.head())
    return jsonify({"message": "Data imputation completed"})


@app.route('/standardize' , methods = ['POST'])
def labelEncode():
    global df
    selected_columns = request.form.get('columns')  # JSON string from the frontend
    selected_columns = json.loads(selected_columns) 

    lbl_encoder=LabelEncoder()
    for column in selected_columns:
        df[column]=lbl_encoder.fit_transform(df[column])
    
    print(df.head())
        
    return jsonify({"message": "Label encoding completed"})

@app.route('/removeoutlier', methods=['POST'])
def removeoutlier():
    global df
    try:
        print("Received request to remove outliers")
        
        if df is None:
            print("Error: No dataframe available")
            return jsonify({"error": "No data available. Please upload a CSV file first."}), 400
            
        # Only process numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            print("Error: No numeric columns found")
            return jsonify({"error": "No numeric columns found in the dataset"}), 400
            
        print(f"Processing {len(numeric_df.columns)} numeric columns")
        
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Create a mask for outliers
        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        outliers_count = outliers.sum()
        
        # Remove outliers from the original dataframe
        df = df[~outliers]
        
        print(f"Removed {outliers_count} outliers. Remaining rows: {len(df)}")
        return jsonify({'Outliers Count': int(outliers_count)}), 200
    except Exception as e:
        print(f"Error in removeoutlier: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/download' , methods = ['GET'])
def download_file():
    print('downloading...')
    global df
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Send the CSV file to the user
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='data.csv'
    )


@app.route('/barplot', methods=['POST'])
def create_barplot():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating bar plot with columns: {selected_columns}")
        
        if not selected_columns:
            # If no columns selected, use the first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns available for plotting'}), 400
            column = numeric_cols[0]
        else:
            # Use the first selected column
            column = selected_columns[0]
            
            # Check if column exists and is numeric
            if column not in df.columns:
                return jsonify({'error': f'Column {column} not found in dataset'}), 400
            
            # Try to convert to numeric if not already
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                pass
        
        plt.figure(figsize=(10, 6))
        df[column].value_counts().sort_index().plot(kind='bar')
        plt.title(f'Bar Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/barplot.png')
        plt.close()
        
        return jsonify({'message': 'Bar plot created successfully', 'plot_url': 'static/barplot.png'}), 200
    except Exception as e:
        print(f"Error creating bar plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/lineplot', methods=['POST'])
def create_lineplot():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating line plot with columns: {selected_columns}")
        
        if not selected_columns:
            # If no columns selected, use the first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns available for plotting'}), 400
            column = numeric_cols[0]
        else:
            # Use the first selected column
            column = selected_columns[0]
            
            # Check if column exists and is numeric
            if column not in df.columns:
                return jsonify({'error': f'Column {column} not found in dataset'}), 400
            
            # Try to convert to numeric if not already
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                pass
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(df)), df[column])
        plt.title(f'Line Plot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/lineplot.png')
        plt.close()
        
        return jsonify({'message': 'Line plot created successfully', 'plot_url': 'static/lineplot.png'}), 200
    except Exception as e:
        print(f"Error creating line plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/scatterplot', methods=['POST'])
def create_scatterplot():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating scatter plot with columns: {selected_columns}")
        
        if len(selected_columns) < 2:
            # If less than 2 columns selected, use the first two numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return jsonify({'error': 'Need at least 2 numeric columns for scatter plot'}), 400
            x_column = numeric_cols[0]
            y_column = numeric_cols[1]
        else:
            # Use the first two selected columns
            x_column = selected_columns[0]
            y_column = selected_columns[1]
            
            # Check if columns exist
            if x_column not in df.columns or y_column not in df.columns:
                return jsonify({'error': 'One or more selected columns not found in dataset'}), 400
            
            # Try to convert to numeric if not already
            try:
                df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
                df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            except:
                pass
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_column], df[y_column])
        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/scatterplot.png')
        plt.close()
        
        return jsonify({'message': 'Scatter plot created successfully', 'plot_url': 'static/scatterplot.png'}), 200
    except Exception as e:
        print(f"Error creating scatter plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap', methods=['POST'])
def create_heatmap():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating heatmap with columns: {selected_columns}")
        
        if not selected_columns:
            # If no columns selected, use all numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                return jsonify({'error': 'Need at least 2 numeric columns for heatmap'}), 400
        else:
            # Use selected columns
            # Try to convert selected columns to numeric
            for col in selected_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            # Filter to only include columns that exist and are numeric
            valid_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(valid_columns) < 2:
                return jsonify({'error': 'Need at least 2 valid numeric columns for heatmap'}), 400
            
            numeric_df = df[valid_columns]
        
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/heatmap.png')
        plt.close()
        
        return jsonify({'message': 'Heatmap created successfully', 'plot_url': 'static/heatmap.png'}), 200
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/violinplot', methods=['POST'])
def create_violinplot():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating violin plot with columns: {selected_columns}")
        
        if not selected_columns:
            # If no columns selected, use the first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns available for plotting'}), 400
            column = numeric_cols[0]
        else:
            # Use the first selected column
            column = selected_columns[0]
            
            # Check if column exists
            if column not in df.columns:
                return jsonify({'error': f'Column {column} not found in dataset'}), 400
            
            # Try to convert to numeric if not already
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except:
                pass
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=df[column])
        plt.title(f'Violin Plot of {column}')
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/violinplot.png')
        plt.close()
        
        return jsonify({'message': 'Violin plot created successfully', 'plot_url': 'static/violinplot.png'}), 200
    except Exception as e:
        print(f"Error creating violin plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/silhouetteplot', methods=['POST'])
def create_silhouetteplot():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get selected columns from request
        data = request.get_json()
        selected_columns = data.get('columns', []) if data else []
        
        print(f"Creating silhouette plot with columns: {selected_columns}")
        
        if not selected_columns:
            # If no columns selected, use all numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                return jsonify({'error': 'Need at least 2 numeric columns for silhouette plot'}), 400
        else:
            # Use selected columns
            # Try to convert selected columns to numeric
            for col in selected_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            # Filter to only include columns that exist and are numeric
            valid_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if len(valid_columns) < 2:
                return jsonify({'error': 'Need at least 2 valid numeric columns for silhouette plot'}), 400
            
            numeric_df = df[valid_columns]
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_samples, silhouette_score
        
        X = StandardScaler().fit_transform(numeric_df)
        
        # Choose a reasonable number of clusters
        n_clusters = min(5, len(df) // 10) if len(df) > 20 else 2
        
        # Apply KMeans clustering
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        # Create the silhouette plot
        plt.figure(figsize=(10, 6))
        y_lower = 10
        
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        plt.title(f"Silhouette Plot for KMeans Clustering\nSilhouette Score: {silhouette_avg:.3f}")
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster label")
        
        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.tight_layout()
        
        # Save plot to a temporary file
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/silhouetteplot.png')
        plt.close()
        
        return jsonify({'message': 'Silhouette plot created successfully', 'plot_url': 'static/silhouetteplot.png'}), 200
    except Exception as e:
        print(f"Error creating silhouette plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({"status": "success", "message": "Connection successful"}), 200

# Add these routes for ML algorithms
@app.route('/linear-regression', methods=['POST'])
def linear_regression():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get data from request
        data = request.get_json()
        features = data.get('features', [])
        target = data.get('target', '')
        test_size = float(data.get('test_size', 0.2))
        
        # Validate inputs
        if not features or not target:
            return jsonify({'error': 'Features and target are required'}), 400
        
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found in dataset'}), 400
        
        # Filter valid feature columns (must exist and be numeric)
        valid_features = []
        for col in features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
        
        if not valid_features:
            return jsonify({'error': 'No valid numeric feature columns selected'}), 400
        
        if not pd.api.types.is_numeric_dtype(df[target]):
            return jsonify({'error': 'Target column must be numeric for regression'}), 400
        
        # Prepare data
        X = df[valid_features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Get coefficients
        coefficients = {feature: coef for feature, coef in zip(valid_features, model.coef_)}
        
        # Create a scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        
        # Save plot
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/regression_plot.png')
        plt.close()
        
        return jsonify({
            'message': 'Linear Regression completed successfully',
            'metrics': {
                'mse': mse,
                'r2': r2,
                'intercept': float(model.intercept_)
            },
            'coefficients': coefficients,
            'plot_url': 'static/regression_plot.png'
        }), 200
        
    except Exception as e:
        print(f"Error in linear regression: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logistic-regression', methods=['POST'])
def logistic_regression():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get data from request
        data = request.get_json()
        features = data.get('features', [])
        target = data.get('target', '')
        test_size = float(data.get('test_size', 0.2))
        
        # Validate inputs
        if not features or not target:
            return jsonify({'error': 'Features and target are required'}), 400
        
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found in dataset'}), 400
        
        # Filter valid feature columns (must exist and be numeric)
        valid_features = []
        for col in features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
        
        if not valid_features:
            return jsonify({'error': 'No valid numeric feature columns selected'}), 400
        
        # Prepare data
        X = df[valid_features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create a confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot
        os.makedirs('static', exist_ok=True)
        plt.savefig('static/confusion_matrix.png')
        plt.close()
        
        return jsonify({
            'message': 'Logistic Regression completed successfully',
            'metrics': {
                'accuracy': accuracy,
                'classification_report': report
            },
            'plot_url': 'static/confusion_matrix.png'
        }), 200
        
    except Exception as e:
        print(f"Error in logistic regression: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/random-forest', methods=['POST'])
def random_forest():
    global df
    if df is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        # Get data from request
        data = request.get_json()
        features = data.get('features', [])
        target = data.get('target', '')
        test_size = float(data.get('test_size', 0.2))
        task_type = data.get('task_type', 'classification')  # 'classification' or 'regression'
        n_estimators = int(data.get('n_estimators', 100))
        
        # Validate inputs
        if not features or not target:
            return jsonify({'error': 'Features and target are required'}), 400
        
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found in dataset'}), 400
        
        # Filter valid feature columns (must exist and be numeric)
        valid_features = []
        for col in features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_features.append(col)
        
        if not valid_features:
            return jsonify({'error': 'No valid numeric feature columns selected'}), 400
        
        # Prepare data
        X = df[valid_features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model based on task type
        if task_type == 'regression':
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Create a scatter plot of actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Random Forest: Actual vs Predicted Values')
            
            # Save plot
            os.makedirs('static', exist_ok=True)
            plt.savefig('static/rf_regression_plot.png')
            plt.close()
            
            # Feature importance
            feature_importance = {feature: importance for feature, importance in zip(valid_features, model.feature_importances_)}
            
            return jsonify({
                'message': 'Random Forest Regression completed successfully',
                'metrics': {
                    'mse': mse,
                    'r2': r2
                },
                'feature_importance': feature_importance,
                'plot_url': 'static/rf_regression_plot.png'
            }), 200
            
        else:  # classification
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Create a confusion matrix
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Random Forest: Confusion Matrix')
            
            # Save plot
            os.makedirs('static', exist_ok=True)
            plt.savefig('static/rf_confusion_matrix.png')
            plt.close()
            
            # Feature importance
            feature_importance = {feature: importance for feature, importance in zip(valid_features, model.feature_importances_)}
            
            return jsonify({
                'message': 'Random Forest Classification completed successfully',
                'metrics': {
                    'accuracy': accuracy,
                    'classification_report': report
                },
                'feature_importance': feature_importance,
                'plot_url': 'static/rf_confusion_matrix.png'
            }), 200
            
    except Exception as e:
        print(f"Error in random forest: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Make sure to create a static directory for saving plots
import os
if not os.path.exists('static'):
    os.makedirs('static')

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
    print("Created 'uploads' directory")


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port to 5001




