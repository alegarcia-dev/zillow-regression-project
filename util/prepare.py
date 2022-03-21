def scale_data(train, validate, test):
    scaler = MinMaxScaler()
    numeric_columns = train.select_dtypes('number').columns
    
    train[numeric_columns] = scaler.fit_transform(train[numeric_columns])
    validate[numeric_columns] = scaler.transform(validate[numeric_columns])
    test[numeric_columns] = scaler.transform(test[numeric_columns])
    
    return train, validate, test