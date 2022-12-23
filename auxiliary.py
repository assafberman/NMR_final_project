from datetime import datetime


def prompt_message(message_str: str):
    """
    Prompts a message with time stamp
    :param message_str: the message to be displayed
    :return:
    """
    print('{}:  {}'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), message_str))


def single_tuple_embedding(shift: float, multiplicity: int, intensity: int, max_multiplicity: int, max_intensity: int):
    """
    A method for embedding a tuple of (shift, multiplicity, intensity) into an embedded vector
    :param shift: the 13C chemical shift to be embedded
    :param multiplicity: the multiplicity of the peak
    :param intensity: the intensity of the peak
    :param max_multiplicity: maximum multiplicity of whole dataset (helps with OneHot embedding)
    :param max_intensity: maximum intensity of whole dataset (helps with OneHot embedding)
    :return: returns the embedded vector
    """
    embedding = [shift]
    for i in range(max_multiplicity):
        embedding.append(1) if i == (multiplicity - 1) else embedding.append(0)
    for i in range(max_intensity):
        embedding.append(1) if i == (intensity - 1) else embedding.append(0)
    return embedding


def input_embedding(p_input: list, max_multiplicity: int, max_intensity: int):
    """
    embeds a spectrum (list of tuples of the form (shift, multiplicity, intensity))
    :param p_input: list representing a spectrum
    :param max_multiplicity: maximum multiplicity of whole dataset (helps with OneHot embedding)
    :param max_intensity: maximum intensity of whole dataset (helps with OneHot embedding)
    :return:an embedding vector of the spectrum. embedding is done peak-wise (tuple-wise).
    """
    vect = []
    for shift, multiplicity, intensity in zip(p_input[::3], p_input[1::3], p_input[2::3]):
        vect.append(single_tuple_embedding(shift, int(multiplicity), int(intensity), int(max_multiplicity), int(max_intensity)))
    return vect


def extract_maximum_multiplicity_intensity(all_spectra_list: list):
    max_multiplicity = 0
    max_intensity = 0
    for element in all_spectra_list:
        max_multiplicity_in_element = max(element[1::3])
        max_intensity_in_element = max(element[2::3])
        if max_multiplicity_in_element > max_multiplicity:
            max_multiplicity = max_multiplicity_in_element
        if max_intensity_in_element > max_intensity:
            max_intensity = max_intensity_in_element
    return int(max_multiplicity), int(max_intensity)
