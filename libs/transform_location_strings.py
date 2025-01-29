def get_renaming_mapping():
    replacements = [
        # Remove unnecessary stuff
        ('_', ' '),
        ('-', ' '),
        ('ctx', ' '),
        ('rh', 'right '),
        ('lh', 'left '),
        ('left ', 'left '),
        ('right ', 'right '),
        ('cerebral', ' '),
        
        # Specific names
        ('ins lg and s cent ins', 'insular long and central sulcus'),
	    ('plan polar', 'planum polare'),
	    ('plan tempo', 'planum temporale'),
	    ('intrapariet and p trans', 'intraparietal and parietal transverse'),
	    ('temp sup g t transv', 'temporal superior anterior transverse'),
        
        # Abbreviations
        (' g ', ' gyrus '),
        (' s ', ' sulcus '),
        (' inf ', ' inferior '),
        (' lat ', ' lateral '),
        (' wm ', 'white matter '), 
        (' ant ', ' anterior '),
        (' cingul ', ' cingulate '), 
        (' front ', ' frontal '),
        (' oc ', ' occipital '),
        (' temp ', ' temporal '),
        (' pariet ', ' parietal '),
        (' fusifor ', ' fusiform'),
        (' med ', ' medial '),
        (' fis ', ' fissure '),
        (' sup ', ' superior '),
        (' post ', ' posterior '),
        (' lat ', ' lateral '),
        (' transv ', ' transverse '),
        (' vent ', ' ventricle '),
        (' supramar ', ' supramarginal '),
        (' horizont ', ' horizontal '),
        (' triangul ', ' triangular '),
        (' parahip ', ' parahippocampal '),
        (' collat ', ' collateral '),
        (' frontomargin ', ' frontomarginal '),
        (' intrapariet ', ' intraparietal '),
            
	    # Remove double spaces
        ('  ', ' '),
    ]

    return replacements

def beautify_str(loc, replace_dict = {}):
    loc = loc.center(len(loc)+2) #add spaces
    loc = loc.lower()

    replacement_mapping = get_renaming_mapping()

    for to_replace, replacement in replacement_mapping:
        loc = loc.replace(to_replace, replacement)
    return loc.strip().capitalize()


if __name__ == '__main__':
    random_test_location_names = ['ctx_lh_S_front_sup', 'ctx_rh_G_postcentral',
       'ctx_lh_G_temp_sup-Lateral', 'ctx_rh_S_temporal_transverse',
       'Left-Inf-Lat-Vent', 'ctx_rh_G_occipital_middle',
       'ctx_rh_Pole_temporal', 'ctx_lh_S_collat_transv_post',
       'Left-Lateral-Ventricle', 'ctx_lh_G_temp_sup-Plan_tempo',
       'ctx_lh_G_oc-temp_lat-fusifor', 'ctx_lh_S_circular_insula_ant',
       'ctx_rh_S_front_middle', 'ctx_lh_S_precentral-inf-part',
       'ctx_lh_G_front_inf-Opercular']

    for location in random_test_location_names:
        print('{:<50s} {:<50s}'.format(location, beautify_str(location)))