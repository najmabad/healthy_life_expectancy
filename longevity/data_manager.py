import pandas as pd
import numpy as np
from tqdm import tqdm


class DataManager:
    """
    DataManager is the class that handles all the data cleaning
    and feature engineering for the share data-set
    """

    def __init__(self, easy_share_dta_path: str, share_death_dta_path: str, verbose: str = True) -> None:
        """
        Initialize the DataManager object
        :param easy_share_dta_path: path to the EasyShare dataset
        :param share_death_dta_path: path to the Share cover screens dataset
        :param verbose: parameter to control whether to print messages
        """
        self.verbose = verbose
        self.df = DataManager.read_dataset(easy_share_dta_path, share_death_dta_path)

    @staticmethod
    def read_dataset(easy_share_dta_path: str, share_death_dta_path: str) -> pd.DataFrame:
        """
        Static method to read the share data-set
        :param easy_share_dta_path: path to the EasyShare data-set
        :param share_death_dta_path: path to the Share cover screens data-set
        :return: a DataFrame containing the share data-set (including deaths from the cover screens)
        """
        # load the two datasets and set indeces
        df = pd.read_stata(easy_share_dta_path)
        df.index = df['mergeid']

        df_deaths = pd.read_stata(share_death_dta_path)
        df_deaths.index = df_deaths['mergeid']

        # filter a subset of columns
        columns = ['mergeid', 'wave', 'country', 'female', 'age', 'adla', 'income_pct_w1', 'income_pct_w2',
                   'income_pct_w4', 'income_pct_w5', 'income_pct_w6', 'int_year', 'thinc_m', 'dn004_mod']

        death_columns = ['deceased_age']

        # apply to datasets
        df = df[columns]
        df_deaths = df_deaths[death_columns]

        # merge the two datasets
        df = pd.merge(df, df_deaths, how='left', left_index=True, right_index=True)
        del df_deaths  # free up memory

        # exclude wave 3 because there is no information about income and disability
        df = df[df['wave'] != 3]
        return df

    def log(self, *messages: str) -> None:
        """
        Print the messages if the instance is verbose
        :param messages: messages to print
        """
        if self.verbose:
            print(*messages)

    def process_country(self):
        """
        Process the country variable of the dataset:
        1) convert to string and remove the country numerical index
        2) filter to a subset of countries
        """
        # convert country into string
        self.df['country'] = self.df['country'].astype('str')

        def convert_country(r):
            r['country'] = r['country'].split()[1]
            return r['country']

        self.df.loc[:, 'country'] = self.df.apply(convert_country, axis=1)

        # keep only a subset of countries
        self.df = self.df[~self.df['country'].isin(['Croatia', 'Czech', 'Estonia', 'Greece', 'Slovenia',
                                                    'Portugal', 'Poland', 'Netherlands', 'Luxembourg',
                                                    'Hungary', 'Ireland', 'Israel'])]

    def process_gender(self) -> None:
        """
        Process the gender variable by renaming it
        """

        def convert_female(r):
            if r['female'] == '0. male':
                return 'male'
            else:
                return 'female'

        self.df['gender'] = self.df.apply(convert_female, axis=1)
        self.df['gender_num'] = (self.df['gender'] == "female").astype(int)

    def process_age(self, start_age: int, end_age: int) -> None:
        """
        Process the age variable by convertint it into
        float and recode missing values
        :param start_age: Minimum age for the analysis
        :param end_age: Maximum age for the analysis
        """

        def convert_age(r):

            if r['age'] == '-15. no information':
                return np.nan
            else:
                return r['age']

        self.df['age'] = self.df.apply(convert_age, axis=1)

        # drop missing values
        missing_age = self.df['age'].isnull().sum()
        self.log('We drop {} observations because there is no information about age'.format(missing_age))
        self.df = self.df.dropna(subset=['age'])

        # filter for individuals aged 50 or above (defined by the 'start_age' variable)
        below_age = (self.df['age'] < start_age).astype('int')
        all_below_age = below_age.sum()
        self.log('We drop {} observations because they are younger than {} years old'.format(all_below_age, start_age))
        self.df = self.df[self.df['age'] >= start_age]

        # drop individuals aged 91 or above (defined by the 'end_age' variable) because of small sample size
        above_age = (self.df['age'] > end_age).astype('int')
        all_above_age = above_age.sum()
        self.log('We drop {} observations because they are older than {} years old'.format(all_above_age, end_age))
        self.df = self.df[self.df['age'] <= end_age]

        self.log('Maximum age:', self.df['age'].max(), 'Minimum age:', self.df['age'].min())

    def process_disability(self) -> None:
        """
        Process the ADLA variable and create a disability
        variable which is 1 if ADLA is greater than 0, and 0 otherwise.
        """
        # convert adla to numeric
        self.df['adla'] = pd.to_numeric(self.df['adla'], errors='coerce')

        # drop missing values
        missing_adla = self.df['adla'].isnull().sum()
        self.log('We drop {} observations because there is no information about adla'.format(missing_adla))
        self.df = self.df.dropna(subset=['adla'])

        # create the dummy 'disabled' if the individual has any functional disability (i.e adla > 0)
        def convert_adla(r):
            if r['adla'] == 0:
                return 0
            if r['adla'] > 0:
                return 1

        self.df.loc[:, 'disabled'] = self.df.apply(convert_adla, axis=1)

    def process_income(self, income_bins: int) -> None:
        """
        Process the income variable by recoding it so that its
        value represents the income decile of the respondent
        during the interview wave
        :param income_bins: maximum number of bins
        """

        def convert_income(r):
            if not str(r['income_pct_w1']).startswith("-13"):
                return r['income_pct_w1']
            if not str(r['income_pct_w2']).startswith("-13"):
                return r['income_pct_w2']
            if not str(r['income_pct_w4']).startswith("-13"):
                return r['income_pct_w4']
            if not str(r['income_pct_w5']).startswith("-13"):
                return r['income_pct_w5']
            if not str(r['income_pct_w6']).startswith("-13"):
                return r['income_pct_w6']

        self.df = self.df.dropna(subset=['thinc_m'])
        self.df['income_dcl'] = self.df.apply(convert_income, axis=1) // (
            (10 // income_bins + 1) if income_bins < 10 else 1)
        # TODO: fix the income binning process
        self.df['income'] = self.df['thinc_m'].astype(float)

        def income_nan(r):
            if r['income'] == '-7. not yet coded':
                return np.nan
            else:
                return r['income']

        self.df.loc[:, 'income'] = self.df.apply(income_nan, axis=1)

        # drop missing values
        self.log(
            'We drop {} observations because there is no information about income'.format(
                self.df['income'].isnull().sum()))
        self.df = self.df.dropna(subset=['income'])

    def process_deceased_age(self) -> None:
        """
        Process the deceased age variable
        """

        def convert_deceased_age(r):
            if r['deceased_age'] == 'Refusal' or r['deceased_age'] == 'Don\'t know':
                return np.nan
            else:
                return r['deceased_age']

        self.df.loc[:, 'deceased_age'] = self.df.apply(convert_deceased_age, axis=1)

        # drop missing values
        missing_deceased_age = self.df['deceased_age'].isnull().sum()
        self.log('We drop {} observations because there is no information about their deceased age'.format(
            missing_deceased_age))
        self.df = self.df.dropna(subset=['deceased_age'])

        # exclude observations with deceased_age < age at interview
        def deceased_age_int(r):
            if r['deceased_age'] == 'Not applicable':
                return 99999
            else:
                return r['deceased_age']

        self.df['deceased_age_int'] = self.df.apply(deceased_age_int, axis=1)

        # transform 'age' into integer to allow comparison
        def age_integer(r):
            return int(round(r['age']))

        self.df['age_int'] = self.df.apply(age_integer, axis=1)

        wrong_deceased_age = self.df['deceased_age_int'] < self.df['age_int']
        self.log('We drop {} observations because their age at death is less than their age at interview'.format(
            wrong_deceased_age.sum()))
        self.df = self.df[self.df['deceased_age_int'] >= self.df['age_int']]

    def process_immigration(self) -> None:
        """
        Process the dn004_mod variable to account for immigration.
        Immigrants will be dropped from the study
        """

        def convert_dn004_mod(r):
            if r['dn004_mod'] == '-15. no information' or r['dn004_mod'] == '-12. don\'t know / refusal':
                return np.nan
            else:
                return r['dn004_mod']

        self.df['born_in_country'] = self.df.apply(convert_dn004_mod, axis=1)

        # drop missing values
        self.log('We drop {} observations because there is no information about where they were born'.format(
            self.df['born_in_country'].isnull().sum()))
        self.df = self.df.dropna(subset=['born_in_country'])
        # drop immigrants
        self.df['born_in_country'] = (self.df['born_in_country'] == '1. Yes').astype('bool')
        immigrants = len(self.df[~self.df['born_in_country']])
        self.log('We drop {} observations because there where not born in the country'.format(immigrants))
        self.df = self.df[self.df['born_in_country']]

    def process_age_of_death(self) -> None:
        """
        Create a dummy to code age of death of individuals and
        create a new variable representing the age of death or at interview
        """

        def deceased_dummy(r):
            if r['deceased_age'] == 'Not applicable':
                return 0
            else:
                return 1

        self.df['is_dead'] = self.df.apply(deceased_dummy, axis=1)

        def deceased(r):
            if r['deceased_age'] == 'Not applicable':
                return r['age_int']
            else:
                return r['deceased_age']

        self.df['is_aged'] = self.df.apply(deceased, axis=1)

    @staticmethod
    def create_panel_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a panel dataframe that can be used for regressions. This
        creates a target y variable corresponding to the death of the
        individual in the following period. This method will also add
        a row a year before the death of the individual with the latest
        available information about the individual if needed
        (i.e. if no record is available for the
        year previous to the death of the individual)
        :param df: a DataFrame resulting to a call to `prepare_dataset`
        :return: a panel DataFrame ready to be used in regression models.
        """
        new_rows = []

        df = df.copy()
        df['y'] = 0

        for mergeid, individual_history in tqdm(df.groupby('mergeid')):
            tot = len(individual_history)
            for i, (idx, row) in enumerate(individual_history.sort_values('age', ascending=True).iterrows()):
                if i < tot - 1 or row['is_dead'] == 0:
                    new_rows.append(row.to_dict())
                    continue
                # last rows of dead people
                if row['deceased_age'] == row['age_int'] + 1:
                    row['y'] = 1
                    new_rows.append(row.to_dict())
                elif row['deceased_age'] == row['age_int']:
                    new_row = row.to_dict()
                    new_row['age'] = row['deceased_age'] - 1
                    new_row['age_int'] = row['deceased_age'] - 1
                    new_row['y'] = 1
                    new_rows.append(new_row)
                else:
                    new_rows.append(row.to_dict())
                    new_row = row.to_dict()
                    new_row['age'] = row['deceased_age'] - 1
                    new_row['age_int'] = row['deceased_age'] - 1
                    new_row['y'] = 1
                    new_rows.append(new_row)

        panel_df = pd.DataFrame(new_rows)
        panel_df = panel_df.sort_values(['mergeid', 'age'])
        return panel_df

    def prepare_dataset(self, start_age: int = 65, end_age: int = 90, income_bins: int = 10) -> pd.DataFrame:
        """
        Pipeline method to call all the data cleaning methods
        of the class and prepare a fully cleaned dataframe.
        :param start_age: Minimum age for the analysis
        :param end_age: Maximum age for the analysis
        :param income_bins: maximum number of bins
        :return: a cleaned and processed dataset for further analysis
        """
        self.process_country()
        self.process_gender()
        self.process_age(start_age, end_age)
        self.process_disability()
        self.process_income(income_bins)
        self.process_deceased_age()
        self.process_immigration()
        self.process_age_of_death()
        self.log('We are left with {} observations'.format(len(self.df)))
        return self.df