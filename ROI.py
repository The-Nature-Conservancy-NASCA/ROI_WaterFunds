# -*- coding: utf-8 -*-
# Import Packages
import os
import pandas as pd
import numpy as np
from numpy.core._multiarray_umath import ndarray


def ROI_Analisys(PathProject_ROI):
    '''
    ####################################################################################################################
                                                    Leer Archivos de entrada
    ####################################################################################################################
    '''
    # Leer Archivos de entrada
    CostFunNBS_Cap  = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','1_CostFunction_NBS_Cap.csv'))
    CostFunBaU_Cap  = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','2_CostFunction_BaU_Cap.csv'))
    CostFunNBS_PTAP = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','3_CostFunction_NBS_PTAP.csv'))
    CostFunBaU_PTAP = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','4_CostFunction_BaU_PTAP.csv'))
    CostNBS         = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','5_NBS_Cost.csv'))
    Porfolio        = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','6_Porfolio_NBS.csv'))
    TD              = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','7_Financial_Parmeters.csv'))
    TimeAnalisys    = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','8_Time.csv'))
    C_BaU           = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','9-OUTPUTS_BaU.csv'))
    C_NBS           = pd.read_csv( os.path.join(PathProject_ROI,'INPUTS','10-OUTPUTS_NBS.csv'))

    # Calculo de Beneficio - Total
    Benefit_Cap     = CostFunBaU_Cap.values[:,2:].T - CostFunNBS_Cap.values[:,2:].T
    Benefit_PTAP    = CostFunBaU_PTAP.values[:,2:].T - CostFunNBS_PTAP.values[:,2:].T

    #CostFunNBS_Cap.values[:,2:].T
    '''
    ####################################################################################################################
                                                    Calculo de inversiones NBS
    ####################################################################################################################
    '''
    del CostNBS['Parameters']
    Porfolio    = Porfolio.set_index('Time')
    NameNBS     = CostNBS.columns
    t_roi       = TimeAnalisys['Time_ROI'][0]
    t_nbs       = TimeAnalisys['Time_Implementation_NBS'][0]
    n_nbs       = len(NameNBS)
    # Costos de implementación
    Cost_I      = np.zeros((t_roi,n_nbs))
    # Costos de manteniemiento y operación
    Cost_M      = np.zeros((t_roi,n_nbs))
    # Costos de oportunidad
    Cost_O      = np.zeros((t_roi,n_nbs))
    # Costos de plataforma
    Cost_P      = np.ones((t_roi, len(TD.values) - 5))

    # Implementation Cost
    Cost_I[0:t_nbs,:]   = Porfolio.values*CostNBS.values[0,:]

    # Costos de Operación y Mantenimiento
    for j in range(0,t_nbs):
        Tmp1 = np.zeros((t_roi, n_nbs))
        for i in range(0,n_nbs):
            Posi = np.arange(j,t_roi,CostNBS[3:].values[0][i])
            Tmp1[Posi,i] = 1

        Cost_M = Cost_M + Tmp1*Porfolio.values[j,:]*CostNBS.values[1,:]

    # Costo de Oportunidad
    Cost_O[0:t_nbs,:]   = np.cumsum(Porfolio.values*CostNBS.values[2,:],0)
    Cost_O[t_nbs:,:]    = Cost_O[t_nbs-1,:]

    # costos de Transacción
    TD = TD.set_index('ID')
    Cost_T = (np.sum(Cost_I,1) + np.sum(Cost_M,1) + np.sum(Cost_O,1))*TD['Value'][1]

    # Costos de Plataforma
    Cost_P = Cost_P*TD['Value'].values[5:]

    # Carbons
    Factor  = 44/12 #44 g/mol CO2 - 12 g/mol C
    Carbons = Factor*(C_NBS['WC (Ton)'] - C_BaU['WC (Ton)'])*TD['Value'][5]
    Carbons = Carbons.values[1:]

    # Total de costo de procesos + Carbono
    TotalBenefit_1    = np.sum(Benefit_Cap,1) + np.sum(Benefit_PTAP,1) + Carbons
    TotalBenefit_1_TD_1 = TotalBenefit_1/((1 + TD['Value'][2])**np.arange(1,31))
    TotalBenefit_1_TD_2 = TotalBenefit_1/((1 + TD['Value'][3])**np.arange(1,31))
    TotalBenefit_1_TD_3 = TotalBenefit_1/((1 + TD['Value'][4])**np.arange(1,31))

    # Total de costo de NBS
    TotalBenefit_2    = Cost_M.sum(1) + Cost_T + Cost_O.sum(1) + Cost_I.sum(1) + Cost_P.sum(1)
    TotalBenefit_2_TD_1  = TotalBenefit_2/((1 + TD['Value'][2])**np.arange(1,31))
    TotalBenefit_2_TD_2  = TotalBenefit_2/((1 + TD['Value'][3])**np.arange(1,31))
    TotalBenefit_2_TD_3  = TotalBenefit_2/((1 + TD['Value'][4])**np.arange(1,31))

    ROI_0 = TotalBenefit_1.sum()/TotalBenefit_2.sum()
    ROI_1 = TotalBenefit_1_TD_1.sum()/TotalBenefit_2_TD_1.sum()
    ROI_2 = TotalBenefit_1_TD_2.sum()/TotalBenefit_2_TD_2.sum()
    ROI_3 = TotalBenefit_1_TD_3.sum()/TotalBenefit_2_TD_3.sum()

    # NPV - I, M, O, T, P
    NPV_I = Cost_I.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, 31))
    NPV_M = Cost_M.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, 31))
    NPV_O = Cost_O.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, 31))
    NPV_T = Cost_T / ((1 + TD['Value'][2]) ** np.arange(1, 31))
    NPV_P = Cost_P.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, 31))

    '''
    ####################################################################################################################
                                                    Guardar Resultados
    ####################################################################################################################
    '''
    NameIndex: ndarray = np.arange(1,t_roi+1)
    Total_1 = pd.DataFrame(data=np.zeros((t_roi,8)),columns=['Cap','PTAP','Carbons','I','M','O','T','P'],index=NameIndex)
    Total_1['Cap']       = np.sum(Benefit_Cap,1)
    Total_1['PTAP']      = np.sum(Benefit_PTAP,1)
    Total_1['Carbons']   = Carbons
    Total_1['I']         = Cost_I.sum(1)
    Total_1['M']         = Cost_M.sum(1)
    Total_1['O']         = Cost_O.sum(1)
    Total_1['T']         = Cost_T
    Total_1['P']         = Cost_P.sum(1)

    Total_2            = pd.DataFrame(data=np.zeros((t_roi,4)),columns=['Total','TD_Min','TD_Mean','TD_Max'])
    Total_2['Total']   = TotalBenefit_1
    Total_2['TD_Min']  = TotalBenefit_1_TD_2
    Total_2['TD_Mean'] = TotalBenefit_1_TD_1
    Total_2['TD_Max']  = TotalBenefit_1_TD_3

    Total_3 = pd.DataFrame(data=np.zeros((t_roi,4)),columns=['Total','TD_Min','TD_Mean','TD_Max'])
    Total_3['Total']   = TotalBenefit_2
    Total_3['TD_Min']  = TotalBenefit_2_TD_2
    Total_3['TD_Mean'] = TotalBenefit_2_TD_1
    Total_3['TD_Max']  = TotalBenefit_2_TD_3

    ROI = pd.DataFrame(data=np.zeros((1,4)),columns=['Total','TD_Min','TD_Mean','TD_Max'])
    ROI['Total']   = ROI_0
    ROI['TD_Min']  = ROI_2
    ROI['TD_Mean'] = ROI_1
    ROI['TD_Max']  = ROI_3

    Cost: Implementation
    Cost: Maintenance
    Cost: Oportunity
    Cost: Transaction
    Cost: Platform

    NPV = pd.DataFrame(data=np.zeros((1, 7)), columns=['Implementation', 'Maintenance', 'Oportunity', 'Transaction', 'Platform', 'Benefit', 'Total'])
    NPV['Implementation']   = -1*NPV_I.sum()
    NPV['Maintenance']      = -1*NPV_M.sum()
    NPV['Oportunity']       = -1*NPV_O.sum()
    NPV['Transaction']      = -1*NPV_T.sum()
    NPV['Platform']         = -1*NPV_P.sum()
    NPV['Benefit']          = TotalBenefit_1_TD_1.sum()
    NPV['Total']            = NPV.sum(1)

    Total_4 = pd.DataFrame(data=Cost_I,columns=NameNBS,index=NameIndex)
    Total_5 = pd.DataFrame(data=Cost_M,columns=NameNBS,index=NameIndex)
    Total_6 = pd.DataFrame(data=Cost_O,columns=NameNBS,index=NameIndex)
    Total_7 = pd.DataFrame(data=Cost_T,columns=['Cost'],index=NameIndex)
    Total_8 = pd.DataFrame(data=Cost_P,columns=TD['Cost'][5:],index=NameIndex)

    Total_9 = CostFunBaU_Cap.groupby(by='Process').sum() - CostFunNBS_Cap.groupby(by='Process').sum()
    del Total_9['Cost_Function']

    Total_10 = CostFunBaU_PTAP.groupby(by='Process').sum() - CostFunNBS_PTAP.groupby(by='Process').sum()
    del Total_10['Cost_Function']

    ROI.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','0_ROI_Sensitivity.csv'))
    Total_1.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','1_GlobalTotals.csv'))
    Total_2.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','2_Benefit_Sensitivity.csv'))
    Total_3.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','3_Cost_Sensitivity.csv'))
    Total_4.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','4_Implementation_Costs.csv'))
    Total_5.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','5_Maintenance_Costs.csv'))
    Total_6.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','6_Opportunity_Costs.csv'))
    Total_7.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','7_Transaction_Costs.csv'))
    Total_8.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','8_Platform_Costs.csv'))
    Total_9.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','9_Cap_Saves.csv'))
    Total_10.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','10_PTAP_Saves.csv'))
    NPV.to_csv(os.path.join(PathProject_ROI,'OUTPUTS','11_NPV.csv'))

# -----------------------------------------------------------------------------------
# Tester
# -----------------------------------------------------------------------------------
PathProject_ROI = r'C:\Users\TNC\Box\01-TNC\28-Project-WaterFund_App\02-Productos-Intermedios\ROI_WaterFunds\Project'
ROI_Analisys(PathProject_ROI)
