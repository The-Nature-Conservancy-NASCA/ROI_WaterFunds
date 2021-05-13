# ROI_WaterFunds
Este Script calcula el análisis ROI haciendo uso de los costos de las funciones de costo de los procesos del sistema de captación y PTAP, así como también, de los costos asociados a las NBS y a los mercados de carbono.

Como parámetros de entra el código requiere los siguientes archivos:

INPUTS
- 1_CostFunction_NBS_Cap.csv
- 2_CostFunction_BaU_Cap.csv
- 3_CostFunction_NBS_PTAP.csv
- 4_CostFunction_BaU_PTAP.csv
- 5_NBS_Cost.csv
- 6_Porfolio_NBS.csv
- 7_Financial_Parmeters.csv
- 8_Time.csv
- 9-CO2_BaU.csv
- 10-CO2_NBS.csv

OUTPUTS
 4 1.0_Implementation_Costs.csv   4 2.2_Maintenance_Costs.csv   4 5.0_Platform_Costs.csv  12 7.2_PTAP_Saves.csv
 4 1.1_Implementation_Costs.csv   4 2.3_Maintenance_Costs.csv   8 5.1_Platform_Costs.csv  12 7.3_PTAP_Saves.csv
 4 1.2_Implementation_Costs.csv   4 3.0_Opportunity_Costs.csv   8 5.2_Platform_Costs.csv   4 8.0_Carbons_Saves.csv
 4 1.3_Implementation_Costs.csv   4 3.1_Opportunity_Costs.csv   8 5.3_Platform_Costs.csv   4 8.1_Carbons_Saves.csv
 4 10_Benefit_Sensitivity.csv     4 3.2_Opportunity_Costs.csv   8 6.0_Cap_Saves.csv        4 8.2_Carbons_Saves.csv
 4 11_Cost_Sensitivity.csv        4 3.3_Opportunity_Costs.csv  12 6.1_Cap_Saves.csv        4 8.3_Carbons_Saves.csv
 1 12_ROI_Sensitivity.csv         4 4.0_Transaction_Costs.csv  12 6.2_Cap_Saves.csv        4 9_GlobalTotals.csv
 1 13_NPV.csv                     4 4.1_Transaction_Costs.csv  12 6.3_Cap_Saves.csv
 4 2.0_Maintenance_Costs.csv      4 4.2_Transaction_Costs.csv   8 7.0_PTAP_Saves.csv
 4 2.1_Maintenance_Costs.csv      4 4.3_Transaction_Costs.csv  12 7.1_PTAP_Saves.csv
