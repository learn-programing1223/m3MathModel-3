import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import optuna

def parse_numeric(val):
    """Parse any numeric value from strings with $, %, commas, etc."""
    if isinstance(val, str):
        num_str = re.sub(r'[^\d.]+', '', val)
        try:
            return float(num_str)
        except ValueError:
            return np.nan
    elif isinstance(val, (int, float)):
        return float(val)
    return np.nan

def main():
    try:
        # Load data
        csv_file = "incomeSheet.csv"
        df = pd.read_csv(csv_file)
        
        print("Dataset loaded with", len(df), "neighborhoods")
        
        # Convert all numeric columns
        numeric_columns = [
            "Approximate Distance(milies)", 
            "Average House Hold Median Income",
            "Estimated Tree Canopy (%)",
            "Number of households",
            "Population",
            "Households with one or more people 65 years and over",
            "Households with one or more people under 18 years",
            "Households with no vehicles",
            "Households with 1+ vehicles"
        ]
        
        # Process each numeric column
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(parse_numeric)
        
        # Advanced Feature Engineering
        # 1. Basic features
        if all(col in df.columns for col in ["Households with one or more people 65 years and over", "Population"]):
            df["pct_elderly"] = (df["Households with one or more people 65 years and over"] / 
                              df["Population"].replace(0, np.nan)) * 100
            df["pct_elderly"] = df["pct_elderly"].fillna(df["pct_elderly"].median())
            df["pct_elderly"] = df["pct_elderly"].clip(0, 100)
            
            # Add square and cube for nonlinear relationships
            df["pct_elderly_sq"] = (df["pct_elderly"]/100)**2 * 100
            df["pct_elderly_cb"] = (df["pct_elderly"]/100)**3 * 100
        
        if all(col in df.columns for col in ["Households with one or more people under 18 years", "Population"]):
            df["pct_under18"] = (df["Households with one or more people under 18 years"] / 
                              df["Population"].replace(0, np.nan)) * 100
            df["pct_under18"] = df["pct_under18"].fillna(df["pct_under18"].median())
            df["pct_under18"] = df["pct_under18"].clip(0, 100)
            
            # Add square for nonlinear relationships
            df["pct_under18_sq"] = (df["pct_under18"]/100)**2 * 100
        
        if all(col in df.columns for col in ["Households with no vehicles", "Number of households"]):
            df["pct_no_vehicle"] = (df["Households with no vehicles"] / 
                                 df["Number of households"].replace(0, np.nan)) * 100
            df["pct_no_vehicle"] = df["pct_no_vehicle"].fillna(df["pct_no_vehicle"].median())
            df["pct_no_vehicle"] = df["pct_no_vehicle"].clip(0, 100)
            
            # Add square for nonlinear relationships
            df["pct_no_vehicle_sq"] = (df["pct_no_vehicle"]/100)**2 * 100
        
        if "Approximate Distance(milies)" in df.columns:
            max_dist = max(df["Approximate Distance(milies)"].max(), 1)
            df["dist_vuln"] = df["Approximate Distance(milies)"] / max_dist
            df["dist_vuln"] = df["dist_vuln"].clip(0, 1)
            
            # Add exponential distance effect
            df["dist_vuln_sq"] = df["dist_vuln"] ** 2
            df["dist_vuln_exp"] = 1 - np.exp(-3 * df["dist_vuln"])  # Exponential falloff
            df["dist_vuln_log"] = np.log1p(df["dist_vuln"] * 9 + 1) / np.log(11)  # Log scaled
        
        if "Average House Hold Median Income" in df.columns:
            max_inc = max(df["Average House Hold Median Income"].max(), 1)
            df["income_vuln"] = 1 - (df["Average House Hold Median Income"] / max_inc)
            df["income_vuln"] = df["income_vuln"].clip(0, 1)
            
            # Add nonlinear income effects
            df["income_vuln_sq"] = df["income_vuln"] ** 2  # Emphasize higher vulnerability
            df["income_vuln_sqrt"] = np.sqrt(df["income_vuln"])  # Emphasize lower vulnerability
            
            # Log transformation of income
            df["log_income"] = np.log1p(df["Average House Hold Median Income"])
            # Income percentile rank (0-1)
            df["income_rank"] = df["Average House Hold Median Income"].rank(pct=True)
        
        if "Estimated Tree Canopy (%)" in df.columns:
            df["tree_canopy_vuln"] = 1 - (df["Estimated Tree Canopy (%)"] / 100)
            df["tree_canopy_vuln"] = df["tree_canopy_vuln"].clip(0, 1)
            
            # Squared effect
            df["tree_canopy_vuln_sq"] = df["tree_canopy_vuln"] ** 2
        
        # 2. Complex interaction terms
        # Income + mobility interaction
        df["income_mobility_index"] = df["income_vuln"] * df["pct_no_vehicle"] / 100
        # Elderly in areas far from hospitals
        df["elderly_distance_risk"] = df["pct_elderly"] * df["dist_vuln"] / 100
        # Children far from services
        df["youth_distance_risk"] = df["pct_under18"] * df["dist_vuln"] / 100
        # Combination of factors
        df["compound_vuln"] = ((df["income_vuln"] + df["pct_no_vehicle"]/100 + df["dist_vuln"]) / 3) ** 2
        
        # 3. Population and density factors
        if all(col in df.columns for col in ["Population", "Number of households"]):
            df["household_size"] = df["Population"] / df["Number of households"].replace(0, np.nan)
            df["household_size"] = df["household_size"].fillna(df["household_size"].median())
            df["household_size"] = df["household_size"].clip(0, 10)
            
            # Add nonlinear household size effects
            df["household_size_sq"] = df["household_size"] ** 2
        
        # Define all possible features
        all_features = [
            # Basic features
            "pct_elderly", "pct_under18", "pct_no_vehicle", "dist_vuln", "income_vuln", "tree_canopy_vuln",
            # Nonlinear transformations
            "pct_elderly_sq", "pct_elderly_cb", "pct_under18_sq", "pct_no_vehicle_sq",
            "dist_vuln_sq", "dist_vuln_exp", "dist_vuln_log", 
            "income_vuln_sq", "income_vuln_sqrt", "log_income", "income_rank",
            "tree_canopy_vuln_sq",
            # Interaction terms
            "income_mobility_index", "elderly_distance_risk", "youth_distance_risk", "compound_vuln",
            # Density factors
            "household_size", "household_size_sq"
        ]
        
        # Filter to only include features that exist in the dataframe
        features = [f for f in all_features if f in df.columns]
        print(f"Using {len(features)} features:", ", ".join(features))
        
        # Ensure no NaN or infinite values
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)
        for col in features:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable - equal weighted sum of basic factors
        basic_features = ["pct_elderly", "pct_under18", "pct_no_vehicle", "dist_vuln", "income_vuln", "tree_canopy_vuln"]
        basic_features = [f for f in basic_features if f in df.columns]
        
        df["vulnerability_score"] = sum(df[col]/100 if "pct_" in col else df[col] for col in basic_features) / len(basic_features)
        df["vulnerability_score"] = df["vulnerability_score"] * 100  # Scale to 0-100
        
        # Prepare data for modeling
        X = df[features].values
        y = df["vulnerability_score"].values
        
        # Scale features - try advanced transformers
        scaler = PowerTransformer(method='yeo-johnson')  # Better than StandardScaler for skewed data
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Hyperparameter optimization using Optuna
        def objective(trial):
            xgb_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': 42
            }
            
            model = XGBRegressor(**xgb_params)
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                r2 = r2_score(y_val, preds)
                cv_scores.append(r2)
            
            return np.mean(cv_scores)
        
        print("Optimizing model parameters...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_params = study.best_params
        print("Best parameters:", best_params)
        
        # Train optimized XGBoost
        best_xgb = XGBRegressor(**best_params)
        best_xgb.fit(X_train, y_train)
        
        # Create ensemble with different models
        base_models = [
            ('xgb', best_xgb),
            ('svr', SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')),
            ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42))
        ]
        
        # Create a stacking regressor
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        )
        
        # Train the stacking model
        print("Training stacked ensemble model...")
        stacking_regressor.fit(X_train, y_train)
        
        # Evaluate the models
        def evaluate_model(name, model, X_train, X_test, y_train, y_test):
            # Training predictions
            train_preds = model.predict(X_train)
            train_r2 = r2_score(y_train, train_preds)
            
            # Test predictions
            test_preds = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_r2 = r2_score(y_test, test_preds)
            
            print(f"\n{name} Results:")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            
            return test_preds, test_r2
        
        # Evaluate both models
        xgb_preds, xgb_r2 = evaluate_model("XGBoost (Optimized)", best_xgb, X_train, X_test, y_train, y_test)
        stack_preds, stack_r2 = evaluate_model("Stacked Ensemble", stacking_regressor, X_train, X_test, y_train, y_test)
        
        # Select the best model
        if stack_r2 > xgb_r2:
            best_model = stacking_regressor
            best_name = "Stacked Ensemble"
            best_preds = stack_preds
            best_r2 = stack_r2
        else:
            best_model = best_xgb
            best_name = "XGBoost"
            best_preds = xgb_preds
            best_r2 = xgb_r2
        
        # Feature importance for XGBoost
        if hasattr(best_xgb, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # Sort features by importance
            importance = best_xgb.feature_importances_
            indices = np.argsort(importance)[::-1]
            sorted_features = [features[i] for i in indices]
            sorted_importance = importance[indices]
            
            # Plot top 15 features
            top_n = min(15, len(sorted_features))
            plt.barh(range(top_n), sorted_importance[:top_n], align='center')
            plt.yticks(range(top_n), [sorted_features[i] for i in range(top_n)])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
            plt.close()
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, best_preds, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), min(best_preds))
        max_val = max(y_test.max(), max(best_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add neighborhood labels
        test_indices = np.array(range(len(df)))[len(y_train):]
        test_neighborhoods = df.iloc[test_indices]["Neighborhood"].values
        
        for i, txt in enumerate(test_neighborhoods):
            plt.annotate(txt, (y_test[i], best_preds[i]), 
                     fontsize=9, alpha=0.7,
                     xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel("Actual Vulnerability Score")
        plt.ylabel("Predicted Vulnerability Score")
        plt.title(f"Actual vs Predicted Vulnerability Scores\n{best_name}: R² = {best_r2:.4f}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png")
        plt.close()
        
        # Make predictions for all neighborhoods
        all_preds = best_model.predict(X_scaled)
        
        # Create vulnerability map data
        vulnerability_data = pd.DataFrame({
            "Neighborhood": df["Neighborhood"].values,
            "Vulnerability Score": all_preds
        })
        
        # Add top contributing factors for each neighborhood
        if hasattr(best_xgb, 'feature_importances_'):
            # Get feature contributions for each neighborhood
            for i, neighborhood in enumerate(df["Neighborhood"]):
                # Calculate weighted contributions for this neighborhood
                contributions = np.zeros(len(features))
                for j, feature in enumerate(features):
                    # Normalize feature value
                    norm_value = (df[feature].iloc[i] - df[feature].min()) / (df[feature].max() - df[feature].min())
                    # Weight by feature importance
                    contributions[j] = norm_value * best_xgb.feature_importances_[j]
                
                # Get top 3 contributing factors
                top_indices = np.argsort(contributions)[::-1][:3]
                vulnerability_data.loc[i, "Top Factor 1"] = features[top_indices[0]]
                vulnerability_data.loc[i, "Top Factor 2"] = features[top_indices[1]] if len(top_indices) > 1 else ""
                vulnerability_data.loc[i, "Top Factor 3"] = features[top_indices[2]] if len(top_indices) > 2 else ""
        
        # Sort by vulnerability score
        vulnerability_data = vulnerability_data.sort_values("Vulnerability Score", ascending=False)
        
        # Save to CSV
        vulnerability_data.to_csv("memphis_vulnerability_ranking.csv", index=False)
        
        print("\nTop 10 most vulnerable neighborhoods:")
        print(vulnerability_data.head(10))
        
        print("\nR² of", best_r2, "achieved with", best_name)
        print("\nFiles saved:")
        print("- feature_importances.png - Shows which features most impact vulnerability")
        print("- actual_vs_predicted.png - Shows model accuracy with neighborhood labels")
        print("- memphis_vulnerability_ranking.csv - Detailed vulnerability scores with top factors")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()