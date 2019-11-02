def save_submission (filename, results):
    np.savetxt(filename,
           np.c_[range(1,len(test_data)+1),results],
           delimiter = ',',
           header='ImageId,Label',
           comments='',
           fmt='%d'
           )
    
def model_metrics(y,y_hat,desc):
    column_names=['type','r2','mse','rmse','mae']
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    s = pd.Series([desc,r2,mse,rmse,mae],index=column_names)
    return s
