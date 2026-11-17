"""Publication metadata shared by comparison reports (formal publication year)."""
NEW_METHODS = ('SANSA', 'PAHNet', 'SAM-SPL', 'MLP-Net')
SUPPLEMENTAL_METHODS = ('MPCNet',)
_FSS = {'Matcher':('ICLR',2024), 'PI-CLIP':('CVPR',2024),
    'ABCDFSS':('CVPR',2024), 'IFA':('CVPR',2024), 'DSV-LFS':('CVPR',2025),
    'TVGTANet':('ACM Multimedia',2025), 'MPA':('CVPR',2026), 'HERA':('CVPR',2026),
    'SANSA':('NeurIPS',2025), 'PAHNet':('ICCV',2025)}
_IR = {'DNANet':('IEEE TIP',2023), 'ISNet':('CVPR',2022), 'MSHNet':('CVPR',2024),
    'SCTransNet':('IEEE TGRS',2024), 'RPCANet':('WACV',2024), 'ILNet':('IEEE TAES',2025),
    'HDNet':('IEEE TGRS',2025), 'MSHNet+NS-FPN':('CVPR',2026), 'PConv+SD':('AAAI',2025),
    'SAM-SPL':('IEEE TGRS',2025), 'SFDTNet':('IEEE TGRS',2025),
    'IOVarNet':('IEEE TGRS',2025), 'MLP-Net':('IEEE TGRS',2025),
    'MPCNet':('IEEE TGRS',2026), 'IRPruneDet':('AAAI',2024), 'IRSAM':('ECCV',2024),
    'IR-SAM2':('Remote Sensing',2026), 'SAMamba':('Information Fusion',2025)}
METADATA = {name:dict(category=category, venue=venue, year=year)
    for category, group in [('FSS',_FSS),('IRSTD',_IR)]
    for name,(venue,year) in group.items()}
METADATA['Ours'] = dict(category='Ours',venue='本文',year=None)
for _removed in ('Matcher','PI-CLIP','DSV-LFS','PConv+SD','SFDTNet','IOVarNet'):
    METADATA.pop(_removed,None)

def metadata(method):
    return METADATA['IFA' if method=='IFA_50label' else method].copy()
