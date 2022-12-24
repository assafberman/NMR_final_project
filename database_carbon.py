from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from rdkit import RDLogger
import tensorflow as tf
import re
from auxiliary import prompt_message, extract_maximum_multiplicity_intensity, input_embedding
import os
from sklearn.preprocessing import StandardScaler


def initialize_dataframe(molecule_list):
    """
    Constructs a pandas DataFrame from a list of molecules (defined by RDKit)
    :param molecule_list: List of RDKit defined molecules
    :return: nmr_df: Dataframe of all molecular properties
    """
    nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in molecule_list if x is not None])
    nmr_df['Name'] = [x.GetProp('_Name') for x in molecule_list if x is not None]
    nmr_df['Smiles'] = [Chem.MolToSmiles(x) for x in molecule_list if x is not None]
    nmr_df['Morgan'] = [morgan_to_list(x) for x in molecule_list if x is not None]
    return nmr_df


def import_nmrshiftdb2_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    return [x for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd')]


def create_proton_carbon_spectra(nmr_df):
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C 0']
    return nmr_df


def trim_dataframe_no_two_spectra(nmr_df):
    """
    Keeps only the records that have both 1H and 13C spectra
    :param nmr_df:
    :return: dataframe containing records that have both 1H and 13C spectra
    """
    return nmr_df[~(pd.isna(nmr_df['Spectrum 13C']))]


def drop_unnecessary_columns(nmr_df):
    """
    Keeps only the relevant columns of the dataframe (Name, SMILES, 1H and 13C spectra)
    :param nmr_df:
    :return:
    """
    return nmr_df.iloc[:, -4:]


def simplify_spectra(nmr_df):
    """
    Convert the string format of the spectra into manageable list
    :param nmr_df:
    :return:
    """
    nmr_df['Spectrum 13C'] = [re.findall(r'^\d+\.?\d*|\|\d+\.?\d*|[a-zA-Z]+', x) for x in nmr_df['Spectrum 13C']]
    nmr_df['Spectrum 13C'] = [list(map(lambda y: re.sub(r'\|', '', y), x)) for x in nmr_df['Spectrum 13C']]
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_shift_multiplicity_association)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_correct_tuples)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_frequency_list)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(aux_num_multiplicity)
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C'].apply(pad_spectrum)
    nmr_df['Input'] = nmr_df['Spectrum 13C']
    return nmr_df


def aux_correct_tuples(spec_list):
    corrected_list = []
    for item in spec_list:
        if isinstance(item, tuple):
            if aux_is_number(item[0]) and item[1].isalpha():
                corrected_list.append(item)
    return corrected_list


def aux_num_multiplicity(spec_list):
    replace_dict = {'S': 1, 's': 1, 'D': 2, 'd': 2, 'T': 3, 't': 3, 'Q': 4, 'q': 4, 'CH': 5, 'u': 6, 'p': 7}
    return [replace_dict[x] if str(x).isalpha() else x for x in spec_list]


def aux_frequency_list(spec_list):
    freq = {}
    freq_list = []
    for item in spec_list:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    for key, val in freq.items():
        if isinstance(key, str):
            freq_list.append(float(key))
        else:
            freq_list.append(float(key[0]))
            freq_list.append(key[1])
        freq_list.append(val)
    return freq_list


def aux_shift_multiplicity_association(spec_list):
    associated_list = []
    for item in zip(spec_list[::2], spec_list[1::2]):
        associated_list.append(item)
    return associated_list


def aux_is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def pad_spectrum(spec_list, size=300):
    lst = np.array(spec_list, dtype=float)
    lst = np.concatenate([lst, np.zeros(size - len(spec_list))], dtype=float)
    return lst.tolist()


def morgan_to_list(molecule):
    fp_list = []
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=512).ToBitString()
    for i in fp:
        fp_list.append(int(i))
    return fp_list


def import_database_as_df():
    RDLogger.DisableLog('rdApp.*')
    nmr_df = initialize_dataframe(import_nmrshiftdb2_database())
    nmr_df = create_proton_carbon_spectra(nmr_df)
    nmr_df = trim_dataframe_no_two_spectra(nmr_df)
    nmr_df = drop_unnecessary_columns(nmr_df)
    nmr_df = simplify_spectra(nmr_df)
    return nmr_df


def import_db_from_pickle(pickle_path='./database/carbon_nmr.pkl'):
    if os.path.exists('./database/carbon_nmr.pkl'):
        prompt_message('Database found. Importing.')
        nmr_df = pd.read_pickle('./database/carbon_nmr.pkl')
    else:
        prompt_message('Importing database from NMRShiftDB2.')
        nmr_df = import_database_as_df()
        nmr_df.to_pickle('./database/carbon_nmr.pkl')
    return nmr_df


def create_input_output_list(nmr_df):
    input_list = [x for x in nmr_df['Input']]
    max_multiplicity, max_intensity = extract_maximum_multiplicity_intensity(input_list)
    input_list = [input_embedding(x, max_multiplicity, max_intensity) for x in input_list]
    output_list = [x for x in nmr_df['Morgan']]
    return input_list, output_list
