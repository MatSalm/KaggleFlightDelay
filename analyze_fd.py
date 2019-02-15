from glob import glob
import pandas as pd
import numpy as np
from numpy import log, mean
from local_config import config
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold

pd.set_option('display.max_columns', 50)
#pd.set_option('display.width',500)
pd.set_option('display.max_rows',100)

top_dest = ['LGA','LAX','DFW']

########## helper functions ##########

def define_failure(x):
	#if x['CANCELLED'] != None:
		#return 1
	if x > 0:
		return 1
	#did not include diverted flights because if it is diverted but on time
	#then it is not a failure
	#did not include  departure delay because it doesnt matter if the plane
	#arrived on time anyways
	else:
		return 0

def estimate_probability(the_X, the_model):
    predicted = the_model.predict_proba(the_X)
    return pd.DataFrame(data=predicted, columns=['prob_default','prob_success'], index=the_X.index)

#calculates log loss
def calc_log_loss(x):
    if x['FAILURE'] == 1:
        return -log(x['prob_success'])
    return -log(1-x['prob_success'])

def format_time(x):
	if pd.isnull(x):
		return np.nan
	else:
		if x == 2400:
			x = 0
		x = "{0:04d}".format(int(x))
		time = datetime.time(int(x[0:2]), int(x[2:4]))
		return time
		

def format_delay(x):
	if pd.isnull(x):
		return 0
	else:
		return x

def define_season(x):
	#spring: mar-may; summer: jun-aug; fall: sept-nov; winter: dec-feb
	if 3 <= x <= 5:
		return 'spr'
	if 6 <= x <= 8:
		return 'sumr'
	if 9 <= x <= 11:
		return 'fal'
	else:
		return 'win'

# def is_spring(x):
# 	if x == 'spr':
# 		return 1
# 	else:
# 		return 0

# def is_summer(x):
# 	if x == 'sumr':
# 		return 1
# 	else:
# 		return 0

# def is_fall(x):
# 	if x== 'fal':
# 		return 1
# 	else:
# 		return 0

# def is_winter(x):
# 	if x=='win':
# 		return 1
# 	else:
# 		return 0

# def is_monday(x):
# 	if x== '1':
# 		return 1
# 	else:
# 		return 0

# def is_tues(x):
# 	if x== '2':
# 		return 1
# 	else:
# 		return 0

# def is_wed(x):
# 	if x== '3':
# 		return 1
# 	else:
# 		return 0

# def is_thu(x):
# 	if x== '4':
# 		return 1
# 	else:
# 		return 0

# def is_fri(x):
# 	if x== '5':
# 		return 1
# 	else:
# 		return 0

# def is_sat(x):
# 	if x== '6':
# 		return 1
# 	else:
# 		return 0

# def is_sun(x):
# 	if x== '7':
# 		return 1
# 	else:
# 		return 0

def top_flights(x):
    if x['DESTINATION_AIRPORT'] in top_dest:
        return 1
    return 0

######################################

def create_df():
	print ('loading data...')
	global create_df
	fList = glob('%s%sflights*.csv' % (config['data_path'], config['slash']))
	print('printing fList',fList)
	dfList = list()
	for f in fList:
		print('loading %s' % (f))
		df = pd.read_csv(f,header=0, index_col= None)
		print(len(df), len(df.columns))
		dfList.append(df)

	df = pd.concat(dfList)

	return df

def clean_df(df):

	print('dropping flights that are not from ORD...')
	NORD = df[df['ORIGIN_AIRPORT'] != 'ORD'].index
	df.drop(NORD, inplace=True)

	print('dropping flights that are not from top destinations...')
	#create column for top flight indicator
	df['TOP_DESTINATION'] = df.apply(top_flights,1)
	NTDEST = df[df['TOP_DESTINATION'] == 0].index
	df.drop(NTDEST, inplace=True)

	print('creating additional column for unique index id...')
	df['ID'] = df['YEAR'].map(str) + df['MONTH'].map(str) + df['DAY'].map(str) + df['AIRLINE'].map(str) + df['FLIGHT_NUMBER'].map(str)

	print('resetting index...')
	df.set_index('ID', inplace=True)

	print('formatting arrival delay...')
	df['DELAY'] = df['ARRIVAL_DELAY'].apply(format_delay,1)

	print('creating additional column to indicate failure...')
	#df['FAILURE'] = df.apply(define_failure,1)
	df['FAILURE'] = df['ARRIVAL_DELAY'].apply(define_failure,1)

	print('creating yes/no columns for seasons...')
	#spring: mar-may; summer: jun-aug; fall: sept-nov; winter: dec-feb
	df['SEASON'] = df['MONTH'].apply(define_season, 1)
	# df['SPRING'] = df['SEASON'].apply(is_spring,1)
	# df['SUMMER'] = df['SEASON'].apply(is_summer,1)
	# df['FALL'] = df['SEASON'].apply(is_fall,1)
	# df['WINTER'] = df['SEASON'].apply(is_winter,1)

	df['SPRING']=np.where(df['SEASON']=='spr', 1,0)
	df['SUMMER']=np.where(df['SEASON']=='sumr', 1,0)
	df['FALL']=np.where(df['SEASON']=='fal', 1,0)
	df['WINTER']=np.where(df['SEASON']=='win', 1,0)

	# print('creating yes/no columns for day of week...')
	# df['MONDAY'] =df['DAY_OF_WEEK'].apply(is_monday,1)
	# df['TUESDAY'] =df['DAY_OF_WEEK'].apply(is_tues,1)
	# df['WEDNESDAY'] =df['DAY_OF_WEEK'].apply(is_wed,1)
	# df['THURSDAY'] =df['DAY_OF_WEEK'].apply(is_thu,1)
	# df['FRIDAY'] =df['DAY_OF_WEEK'].apply(is_fri,1)
	# df['SATURDAY'] =df['DAY_OF_WEEK'].apply(is_sat,1)
	# df['SUNDAY'] =df['DAY_OF_WEEK'].apply(is_sun,1)

	#code below not working
	df['MONDAY']=np.where(df['DAY_OF_WEEK']==1, 1,0)
	df['TUESDAY']=np.where(df['DAY_OF_WEEK']==2, 1,0)
	df['WEDNESDAY']=np.where(df['DAY_OF_WEEK']==3, 1,0)
	df['THURSDAY']=np.where(df['DAY_OF_WEEK']==4, 1,0)
	df['FRIDAY']=np.where(df['DAY_OF_WEEK']==5, 1,0)
	df['SATURDAY']=np.where(df['DAY_OF_WEEK']==6, 1,0)
	df['SUNDAY']=np.where(df['DAY_OF_WEEK']==7, 1,0)

	print('creating yes/no columns for airlines...')
	#creating separate columns for airlines
	df['A_UA']=np.where(df['AIRLINE']=='UA', 1,0)
	df['A_AA']=np.where(df['AIRLINE']=='AA', 1,0)
	df['A_US']=np.where(df['AIRLINE']=='US', 1,0)
	df['A_F9']=np.where(df['AIRLINE']=='F9', 1,0)
	df['A_B6']=np.where(df['AIRLINE']=='B6', 1,0)
	df['A_OO']=np.where(df['AIRLINE']=='OO', 1,0)
	df['A_AS']=np.where(df['AIRLINE']=='AS', 1,0)
	df['A_NK']=np.where(df['AIRLINE']=='NK', 1,0)
	df['A_WN']=np.where(df['AIRLINE']=='WN', 1,0)
	df['A_DL']=np.where(df['AIRLINE']=='DL', 1,0)
	df['A_EV']=np.where(df['AIRLINE']=='EV', 1,0)
	df['A_HA']=np.where(df['AIRLINE']=='HA', 1,0)
	df['A_MQ']=np.where(df['AIRLINE']=='MQ', 1,0)
	df['A_VX']=np.where(df['AIRLINE']=='VX', 1,0)

	print('creating columns for destination indicator...')
	df['LGA']=np.where(df['DESTINATION_AIRPORT']=='LGA', 1,0)
	df['LAX']=np.where(df['DESTINATION_AIRPORT']=='LAX', 1,0)
	df['DFW']=np.where(df['DESTINATION_AIRPORT']=='DFW', 1,0)

	df=df[['DELAY','WINTER','SPRING','SUMMER','FALL','LGA','LAX','DFW','FAILURE','DESTINATION_AIRPORT','MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY','A_UA','A_AA','A_US','A_F9','A_B6','A_OO','A_AS','A_NK','A_WN','A_DL','A_EV','A_HA','A_MQ','A_VX']]

	# print('formatting departure time...')
	# df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_time)
	# df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_time)
	# df['ARRIVAL_TIME'] = df['ARRIVAL_TIME'].apply(format_time)


	#print('creating column counting how many times a plane flies per day...')
	#df['DAY_PLANE'] = df['YEAR'].map(str) + df['MONTH'].map(str) + df['DAY'].map(str) + df['AIRLINE'].map(str) + df['TAIL_NUMBER'].map(str)
	#creating column to count how many times a plane flies per day by finding duplicates
	#df['FPD_COUNT'] = df.groupby(df.DAY_PLANE,as_index=False).size()
	#drop day_plane column we dont need it anymore
	#df = df.drop('DAY_PLANE', 1)

	#df=df.dropna(inplace=True)
	return df

#def analysis():

	#finding top routes
	#count number of destination airports and rank descending
	#df_topdest = df_ordflights.groupby('DESTINATION_AIRPORT').size()
	#df_topdest.sort_values(ascending=False)

def normalize_df(df,features):
    scaler = StandardScaler().fit(df[features])
    df[features] = StandardScaler().fit_transform(df[features]) #features is a predefined list
    return df, scaler

def split_df(df):
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df, df['FAILURE'], test_size=0.2, stratify=df['FAILURE'], random_state=0) #0 to ensure repeatability; None for a different state each time
    return df_X_train, df_X_test, df_y_train, df_y_test

def prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, c=100, use_cv=False, coef_df=pd.DataFrame()):
    #define and create model without cross validation
    if use_cv:
        model = LogisticRegressionCV(tol=1.0e-4, penalty='l2', Cs=25, fit_intercept=True, n_jobs=5, cv=StratifiedKFold(n_splits=5),scoring='neg_log_loss', solver='liblinear', refit=True, random_state=0)
    else:
        model = LogisticRegression(tol=1.0e-4, penalty='l2', C=c, fit_intercept=True, warm_start=True, solver='liblinear')

    model.fit(df_X_train[features], df_y_train)

    #use model to make in-sample and out-of-sample predictions
    df_is = estimate_probability(df_X_train[features],model) #calculating probabilities in model
    df_X_train = pd.concat([df_X_train,df_is], axis=1, join='outer')
    df_X_train['log_loss'] = df_X_train.apply(calc_log_loss,1)
    log_loss_is = mean(df_X_train.log_loss.values)
    #same as above but for testing:
    df_oos = estimate_probability(df_X_test[features],model)
    df_X_test = pd.concat([df_X_test,df_oos], axis=1, join='outer') #all data plus prediction
    df_X_test['log_loss'] = df_X_test.apply(calc_log_loss,1)
    log_loss_oos = mean(df_X_test.log_loss.values)

    # add coefficients + more data to dataframe & label them
    if use_cv: c = model.C_[0]
    coef_df = coef_df.append(pd.Series([c,log_loss_is,log_loss_oos,model.intercept_[0]] + model.coef_[0].tolist()),ignore_index=True)
    coef_df.columns = ['c','log_loss_is','log_loss_oos','intercept'] + features

    return df_X_train, df_X_test, coef_df


def run(use_cv=False):

    features = ['WINTER','SPRING','SUMMER','FALL','LGA','LAX','DFW','MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY','A_UA','A_AA','A_US','A_F9','A_B6','A_OO','A_AS','A_NK','A_WN','A_DL','A_EV','A_HA','A_MQ','A_VX']
    df = create_df()
    df = clean_df(df)
    df, scaler = normalize_df(df,features)
    df_X_train, df_X_test, df_y_train, df_y_test = split_df(df) #splitting df
    df_X_train, df_X_test, coef_df = prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, use_cv=use_cv, c=100)

    return coef_df

if __name__ == '__main__':
    coef_df = run()
