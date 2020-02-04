#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from postal.expand import expand_address
from postal.parser import parse_address

# expand the address, i.e., eliminate abbreviations and use of custom writing styles
# example 'Nineteen' -> 19      'Str.' -> 'Street'
def expand_address_value(address_field):
    expanded_address = expand_address(str(address_field), languages = ['en'])
    
    if len(expanded_address) == 1:
        return expanded_address[0]
    else:
        return address_field


# method to split given values of address into entities
# example: 'Schlossplatz 18 Münster' -> ('road': 'Schlossplatz'), ('house number': '18'), ('city', 'Münster')
def normalize_addr(entry):
    
    addr_to_parse =  entry['exp_addr']     
    x = parse_address(
        addr_to_parse,
        # adapt address parsing to known language and country of known 
        # for improvided parsing
        language='en', 
        country='us')
    
    for val_combo in x:
        column_val = val_combo[0]
        column_name = val_combo[1]
        full_col_name = 'pypost_' + column_name
        entry[full_col_name] = column_val
        
    return entry    


#method to enhance basic customer data set of data samples 
# with additional information gathered with pypostal
def add_new_pp_info(entry):
    
    entry['street'] = None
    entry['house_number'] = None
    entry['house'] = None
    
    entry['address'] = entry['exp_addr']
    
    if entry['pypost_road'] is not None:
        entry['house_number'] = entry['pypost_house_number']
        entry['house'] = entry['pypost_house']
        entry['street'] = entry['pypost_road']

    
    fields_of_interest = [
        'Id',
        'source',
        'name',
        'category',
        'phone',
        'city',
        'address',
        'street',
        'house_number',
        'house',

    ]    
    
    return entry[fields_of_interest]

