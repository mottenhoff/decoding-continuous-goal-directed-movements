from matplotlib import colormaps

def ppt_id():
    return {'kh036': 'p1', 
            'kh040': 'p2',
            'kh041': 'p3',
            'kh042': 'p4',
            'kh043': 'p5',
            'kh045': 'p6',
            'kh046': 'p7',
            'kh047': 'p8',
            'kh048': 'p9',
            'kh049': 'p10', 
            'kh050': 'p11',
            'kh051': 'p12',
            'kh052': 'p13',
            'kh053': 'p14',
            'kh054': 'p15',
            'kh055': 'p16'}

def cmap():
    cmap = colormaps['tab20']
    ppts = ppt_id().values()
    return dict(zip(ppts, [cmap(i) for i, _ in enumerate(ppts)]))