from datetime import datetime


def prompt_message(message_str: str):
    print('{}:  {}'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), message_str))


def aux_vect_shift(shift: float):
    vect = []
    integral = int(shift)
    fractional = shift - integral
    for i in range(8):
        vect.append(int(integral % 2))
        integral /= 2
    vect = vect[::-1]
    for i in range(22):
        fractional *= 2
        if int(fractional):
            vect.append(1)
            fractional -= 1
        else:
            vect.append(0)
    return vect


def aux_vect_multiplicity(multiplicity: int):
    vect = []
    for i in range(3):
        vect.append(int(multiplicity % 2))
        multiplicity /= 2
    return vect[::-1]


def aux_vect_intensity(intensity: int):
    vect = []
    for i in range(5):
        vect.append(int(intensity % 2))
        intensity /= 2
    return vect[::-1]


def single_tuple_embedding(shift: float, multiplicity: int, intensity: int):
    return [shift] + \
           aux_vect_multiplicity(multiplicity) + \
           aux_vect_intensity(intensity)


def input_embedding(p_input: list):
    vect = []
    for shift, muliplicity, intensity in zip(p_input[::3], p_input[1::3], p_input[2::3]):
        vect.append(single_tuple_embedding(shift, muliplicity, intensity))
    return vect
