import numpy as np

framerates = [  60.81644169407352,
                41.95565535512997,
                76.55799366305972,
                66.59473359665783,
                77.62404161898776,
                45.696156184590336,
                74.50907165260432,
                75.11355628258688,
                49.47502701297592,
                74.05126239061477,
                73.1180536286137,
                88.69577283722212,
                37.65368526122749,
                51.92245006240499,
                69.90882578887003,
                75.16204266679118,
                41.31799817774512,
                67.4387842384491,
                43.152366221832644,
                44.16558270028943,
                73.9682764246213,
                76.85805361827808,
                79.3257459572814,
                72.09084313524178,
                72.22689690724309,
                63.64122606877966,
                46.07138823642967,
                72.44089871829281,
                74.227152888682,
                72.22246880235295,
                76.10255121641396,
                66.94010447380964,
                46.87625725210706,
                79.0889339971764,
                79.22271722295122,
                76.82418963245216]

print(np.mean(framerates), np.std(framerates))

total_times = [244.52830150071532,
                292.05859069898725,
                363.0611701980233,
                363.01273219892755,
                444.5511366999708,
                460.0267594999168,
                430.5567000000001,
                505.0528816999995,
                413.085322299999,
                706.5587919000536,
                599.5561312986538,
                634.5602366002277,
                1204.58467563241,
                916.5104705999984,
                1325.7012831000002,
                1057.0642310995609]

print(np.mean(total_times), np.std(total_times), sum(total_times), min(total_times), max(total_times))


total_targets = [50,
                50,
                45,
                18,
                100,
                48,
                100,
                100,
                150,
                27,
                47,
                100,
                150,
                86,
                145,
                296
]

print(np.mean(total_targets), np.std(total_targets), sum(total_targets), min(total_targets), max(total_targets))