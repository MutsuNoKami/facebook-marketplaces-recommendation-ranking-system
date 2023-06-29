import pandas as pd

def clean_price(price_column):
    """removes the currency signs from the file so it is seen as pure numerical data"""
    return price_column.str.replace('[^0-9.]', '', regex=True).astype('float64')

def clean_time(date_col):

    return pd.to_datetime(date_col, infer_datetime_format=True, errors='coerce').dt.strftime('%d/%m/%Y')

def get_tabular_data(filepath, line_term=','):

    filepath = "r'C:\Users\Admin\Documents\FBmarketplace\Products.csv'"
    df = pd.read_csv(filepath_or_buffer=filepath, lineterminator=line_term).dropna()
    
    return df


def text_split(column, character):

    return column.apply(lambda x: x.split(character)[0].lower())

def clean_text_data(column, keep_char=None):
    """The function removes all non-alphanumeric and whitespace characters from the text and allows us to keep a certain number of words 

    """
    non_alpha_numeric = column.str.replace('\W', ' ', regex=True).apply(lambda x: x.lower()) # all characters in lower case
    non_whitespace = non_alpha_numeric.str.replace('\s+', ' ', regex=True)
    # remove all single characters
    clean_text = non_whitespace.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    # Only when we specify the number of words we want to keep, we run this code
    if keep_char != None:
        return clean_text.apply(lambda x: ' '.join(x.split(' ')[0:keep_char])) 
    return clean_text


def get_data_pipeline():
    """This function acts a pipeline where all the previous functions are run to return a cleaned dataset 
    Returns:
        pd.DataFrame: A clean product dataset is returned for further analysis
    """

    data = get_tabular_data('Products.csv', '\n').iloc[:, 1:] 

    data.drop(columns=['url', 'page_id'], inplace=True)

    data['price'] = clean_price(data['price'])

    data.rename(columns={'create_time\r': 'create_time'}, inplace=True)
    data['create_time'] = clean_time(data['create_time'])

    data['location'] = data['location'].astype('category')
    data['category'] = data['category'].astype('category')

    data['product_name'] = clean_text_data(data['product_name'], 10) 
    data['product_description'] = clean_text_data(data['product_description'])


    data.drop_duplicates(subset=['product_name', 'location', 'product_description', 'create_time', 'price'], keep='first', inplace=True)

    data.rename(columns={'id': 'product_id'}, inplace=True)

    return data

if __name__ == '__main__':
    get_data_pipeline()