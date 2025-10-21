import pandas as pd
df = pd.DataFrame({
    'PRODUCTO': ['51030060060201']*8 + ['110000140080001']*10,
    'AÑO': [2023]*4 + [2024]*4 + [2023]*5 + [2024]*5,
    'SEMANA': [7,9,10,12,13,14,15,16] + [1,2,3,4,5,6,7,8,9,10],
    'CANTIDAD': [2034,2486,1760,2926,2244,2500,2100,2300] + [500,600,550,580,590,610,620,630,640,650]
})
df.to_excel("ejemplo_series.xlsx", index=False)
print("ejemplo_series.xlsx creado")
