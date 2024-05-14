from matplotlib import colormaps

# def ppt_id():
#     return {'kh036': 'p1', 
#             'kh040': 'p2',
#             'kh041': 'p3',
#             'kh042': 'p4',
#             'kh043': 'p5',
#             'kh045': 'p6',
#             'kh046': 'p7',
#             'kh047': 'p8',
#             'kh048': 'p9',
#             'kh049': 'p10', 
#             'kh050': 'p11',
#             'kh051': 'p12',
#             'kh052': 'p13',
#             'kh053': 'p14',
#             'kh054': 'p15',
#             'kh055': 'p16'}

def ppt_id():
    return {'kh036': 'sub-01', 
            'kh040': 'sub-02',
            'kh041': 'sub-03',
            'kh042': 'sub-04',  #
            'kh043': 'sub-05',
            'kh045': 'sub-06',
            'kh046': 'sub-07',
            'kh047': 'sub-08',
            'kh048': 'sub-09',
            'kh049': 'sub-10',  #
            'kh050': 'sub-11',
            'kh051': 'sub-12',  #
            'kh052': 'sub-13',
            'kh053': 'sub-14',
            'kh054': 'sub-15',  #
            'kh055': 'sub-16',
            'kh056': 'sub-17',
            'kh067': 'sub-18'}


def cmap():
    cmap = colormaps['tab20']
    ppts = ppt_id().values()
    return dict(zip(ppts, [cmap(i) for i, _ in enumerate(ppts)]))