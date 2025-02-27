import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class FeaturesHandler: 

    X_train = None
    binary_features = []
    ordinal_features = []
    low_card_nominal = []
    high_card_features = []
    tbd_features = []
    ordinal_features_cols = ['Alley', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
        'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

    ordinal_mappings = {
        # Alley access
        'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2},
        
        # Lot shape
        'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
        
        # Utilities
        'Utilities': {'NoSeWa': 0, 'NoSewr': 1, 'AllPub': 3},
        
        # Land Slope
        'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
        
        # Quality ratings (consistent pattern across several features)
        'ExterQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'ExterCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'HeatingQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'PoolQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        
        # Basement exposure
        'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        
        # Basement finished types
        'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        
        # Home functionality
        'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        
        # Garage finish
        'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        
        # Fence quality
        'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    }



    def __init__(self, X_train: pd.DataFrame):
        self.X_train = X_train
    
    def handle_categories(self):
        self.categorise_features()
        self.process_binary_features()
        self.process_ordinal_features()
        self.process_low_card_nominal()
        self.process_tbd_features()
        return self.X_train

    def process_binary_features(self):
        """
        Modify binary descriptive features to be 0 or 1 
        """
        binary_mappings = {
            'Street': {'Pave': 1, 'Grvl': 0},
            'CentralAir': {'Y': 1, 'N': 0}
        }
        for feature, mapping in binary_mappings.items():
            self.X_train[feature] = self.X_train[feature].map(mapping)
    
    # Ensure any unexpected values (if they appear) are handled
        if self.X_train[feature].isna().any():
            print(f"Warning: Unexpected values in {feature} - filling with mode")
            self.X_train[feature] = self.X_train[feature].fillna(self.X_train[feature].mode()[0])

        # for feature in self.binary_features:
        #     print(f"{feature} after encoding: {self.X_train[feature].value_counts().to_dict()}") 
            
    def process_ordinal_features(self):
        """
        Modify ordinal descriptive features to numerical values
        """
        # for each entry in ordinal_features, apply the mapping
        for feature in self.ordinal_features:
            self.X_train[feature] = self.X_train[feature].map(self.ordinal_mappings[feature])
            if self.X_train[feature].isna().any():
                print(f"Warning: Unexpected values in {feature} - filling with mode")
                self.X_train[feature] = self.X_train[feature].fillna(self.X_train[feature].mode()[0])
            # print(f"{feature} after encoding: {X_train[feature].value_counts().to_dict()}")
        
    def process_low_card_nominal(self):
        # Handle missing values first
        for feature in self.low_card_nominal:
            self.X_train[feature] = self.X_train[feature].fillna('NA')
        
        # Create and fit the encoder
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoder.fit(self.X_train[self.low_card_nominal])
        
        # Transform the data
        encoded_features = encoder.transform(self.X_train[self.low_card_nominal])
        encoded_column_names = encoder.get_feature_names_out(self.low_card_nominal)
        
        # Create a dataframe with the encoded features
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoded_column_names,
            index=self.X_train.index
        )
        
        # Drop original columns and concatenate the encoded ones
        self.X_train = self.X_train.drop(columns=self.low_card_nominal)
        self.X_train = pd.concat([self.X_train, encoded_df], axis=1)
        
    def process_high_card_features(self):
        pass
    def process_tbd_features(self):
        y_train = pd.read_csv('data/train.csv')['SalePrice']
        for feature in self.tbd_features:
            # Calculate mean target value for each category
            target_means = self.X_train.join(y_train).groupby(feature)[y_train.name].mean()
        
            # Replace categories with their target mean
            self.X_train[feature] = self.X_train[feature].map(target_means)
       


    def categorise_features(self):
        """
        Categorise features into groups for further processing
        """
        cat_cols = self.X_train.select_dtypes(include=['object']).columns
        # print(f"Cat cols: {cat_cols}")
        cat_info = {}


        for col in cat_cols:    
            unique_vals =  self.X_train[col].value_counts(dropna=False).shape[0]
            missing_vals = self.X_train[col].isna().sum()
            sample_vals = self.X_train[col].value_counts().head(3).index.tolist()
            
            cat_info[col] = {
                'unique_values': unique_vals,
                'missing_values': missing_vals,
                'examples': sample_vals
            }


        for col, info in cat_info.items():
            if col in self.ordinal_features_cols:
                self.ordinal_features.append(col)
            elif info['unique_values'] == 2:
                self.binary_features.append(col)
            elif info['unique_values'] <= 10:
                self.low_card_nominal.append(col)
            elif info['unique_values'] > 10 and info['unique_values'] <= 25:
                self.tbd_features.append(col)       
            elif info['unique_values'] > 25:
                self.high_card_features.append(col)

        # print (f"\nBinary features: {self.binary_features}\n")
        # print (f"Ordinal features: {self.ordinal_features}\n")
        # print (f"Low cardinality nominal features: {self.low_card_nominal}\n")
        # print (f"High cardinality features: {self.high_card_features}\n")
        print (f"Features to be determined: {self.tbd_features}\n")

class FeaturesScaler: 
    def __init__(self, X_train: pd.DataFrame):
        self.X_train = X_train
    
    def scale_features(self):
        scaler = StandardScaler()
        id_train = self.X_train['Id'].copy()
        X_train_no_id = self.X_train.drop('Id', axis=1)
        X_train_scaled_array = scaler.fit_transform(X_train_no_id)
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train_no_id.columns)
        X_train_scaled['Id'] = id_train.values
        return X_train_scaled  
          
def pre_process_numerical_features(X_train):
    # find int floal columns
    int_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"Int cols: {int_cols}")
    # find columns with missing values
    missing_cols = int_cols[X_train[int_cols].isnull().any()]
    print(f"Missing cols: {missing_cols}")
    # fill missing values with mean
    X_train[missing_cols] = X_train[missing_cols].fillna(X_train[missing_cols].mean())
    return X_train


def pre_process_dataset(X_train):
    #Fix column to be treated as int
    X_train["LotFrontage"] = pd.to_numeric(X_train["LotFrontage"], errors='coerce')
    X_train["MasVnrArea"] = pd.to_numeric(X_train["LotFrontage"], errors='coerce')
    X_train["GarageYrBlt"] = pd.to_numeric(X_train["LotFrontage"], errors='coerce')

    # Handle features 
    handler = FeaturesHandler(X_train)
    X_train = handler.handle_categories()
    print(f"output dataframe:\n {X_train.head()}")

    # Scale features 
    scaler = FeaturesScaler(X_train)
    X_train = scaler.scale_features()
    print(f"output dataframe after scaling:\n {X_train.head()}")
    return X_train



def main():
    # train = pd.read_csv('data/train.csv') 

    # X_train = train.drop(columns=['SalePrice'])
    # X_train = pre_process_dataset(X_train)
    # X_train.to_csv('data/X_train_cleaned.csv', index=False)

    # first clean numerical data and save it
    test_set = pd.read_csv('data/test.csv',keep_default_na=True)
    X_test_cleaned_numerical = pre_process_numerical_features(test_set)
    X_test_cleaned_numerical.to_csv('data/X_test_cleaned_numerical.csv', index=False)

    test_set = pd.read_csv('data/X_test_cleaned_numerical.csv',keep_default_na=False)
    X_test = pre_process_dataset(test_set)
    X_test.to_csv('data/X_test_cleaned.csv', index=False)

if __name__ == "__main__":
    main()