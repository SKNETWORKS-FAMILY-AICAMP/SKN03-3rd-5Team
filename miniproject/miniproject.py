from functions import *

df = csv_to_dataframe("miniproject.csv")

dropped_df = drop_column(df, ['customerID'])
encoded_df = transform_columns(dropped_df)

check_class_percentages(encoded_df)

X, y = data_split2(encoded_df, 'Churn')

X_train, X_test, y_train, y_test = data_split(encoded_df, 'Churn')

# 로지스틱 모델
logistic_y_pred, log_model = logistic(X_train, X_test, y_train, y_test)

view_confusion_matrix(log_model, y_test, logistic_y_pred)

evaluate_model_with_threshold(log_model, X, y, 0.3)

y_corrected = change_target(log_model, 0.3, X, y)

X_train, X_test, y_train, y_test = data_split3(X, y_corrected)

visualize_individual_tree(X_train, X_test, y_train, y_test, tree_index=0, max_depth=3)
