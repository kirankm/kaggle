import re
def get_datepart(df, fieldName, drop = True, time = False):
    temp = re.sub("[Dd]ate$","",fieldName) +"_"
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',"weekofyear",'Is_month_end', 'Is_month_start', 
         'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    field = df[fieldName]
    for n in attr:
        df[temp+n] = getattr(field.dt, n.lower())
    df[temp+"Elapsed"] = field.astype(np.int64)
    if drop:
        df.drop(fieldName, axis =1, inplace = True)
