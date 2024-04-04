# -*- coding: utf-8 -*-
# Import Packages
import os
import pandas as pd
import numpy as np
from numpy.core._multiarray_umath import ndarray

INPUTS = 'in'
OUTPUTS = 'out'

def ROI_Analisys(PathProject_ROI):
    '''
    ####################################################################################################################
                                                    Leer Archivos de entrada
    ####################################################################################################################
    '''

    # Leer Archivos de entrada
    CostNBS         = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'5_NBS_Cost.csv'))
    Porfolio        = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'6_Porfolio_NBS.csv'))
    TD              = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'7_Financial_Parmeters.csv'))
    TimeAnalisys    = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'8_Time.csv'))
    C_BaU           = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'9-CO2_BaU.csv'))
    C_NBS           = pd.read_csv( os.path.join(PathProject_ROI,INPUTS,'10-CO2_NBS.csv'))

    Value = os.path.exists(os.path.join(PathProject_ROI, INPUTS, '1_CostFunction_NBS_Cap.csv'))
    if Value:
        CostFunNBS_Cap = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '1_CostFunction_NBS_Cap.csv'))
        CostFunBaU_Cap = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '2_CostFunction_BaU_Cap.csv'))

        Tmp = CostFunBaU_Cap.columns
        if Tmp.shape != (TimeAnalisys['Time_ROI'][0] + 2):
            NameC = ['Process','Cost_Function']
            for ki in range(1,TimeAnalisys['Time_ROI'][0] + 1):
                NameC.append(ki)
            Tmp = pd.DataFrame(columns=NameC, index=[0])
            Tmp['Process'] = 'NoData'
            Tmp['Cost_Function'] = 0
            for ki in range(1, TimeAnalisys['Time_ROI'][0] + 1):
                Tmp[ki] = 0

            CostFunNBS_Cap = Tmp
            CostFunBaU_Cap = Tmp

        Value1 = os.path.exists(os.path.join(PathProject_ROI, INPUTS, '3_CostFunction_NBS_PTAP.csv'))
        if Value1:
            CostFunNBS_PTAP = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '3_CostFunction_NBS_PTAP.csv'))
            CostFunBaU_PTAP = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '4_CostFunction_BaU_PTAP.csv'))

            Tmp = CostFunNBS_PTAP.columns
            if Tmp.shape != (TimeAnalisys['Time_ROI'][0] + 2):
                NameC = ['Process', 'Cost_Function']
                for ki in range(1, TimeAnalisys['Time_ROI'][0] + 1):
                    NameC.append(ki)
                Tmp = pd.DataFrame(columns=NameC, index=[0])
                Tmp['Process'] = 'NoData'
                Tmp['Cost_Function'] = 0
                for ki in range(1, TimeAnalisys['Time_ROI'][0] + 1):
                    Tmp[ki] = 0

                CostFunNBS_PTAP = Tmp
                CostFunBaU_PTAP = Tmp

        else:
            CostFunNBS_PTAP = CostFunNBS_Cap * 0
            CostFunNBS_PTAP['Process'] = 'PTAP_NoData'
            CostFunNBS_PTAP['Cost_Function'] = 0
            CostFunBaU_PTAP = CostFunNBS_PTAP
    else:
        CostFunNBS_PTAP = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '3_CostFunction_NBS_PTAP.csv'))
        CostFunBaU_PTAP = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '4_CostFunction_BaU_PTAP.csv'))

        Tmp = CostFunNBS_PTAP.columns
        if Tmp.shape != (TimeAnalisys['Time_ROI'][0] + 2):
            NameC = ['Process', 'Cost_Function']
            for ki in range(1, TimeAnalisys['Time_ROI'][0] + 1):
                NameC.append(ki)
            Tmp = pd.DataFrame(columns=NameC, index=[0])
            Tmp['Process'] = 'NoData'
            Tmp['Cost_Function'] = 0
            for ki in range(1, TimeAnalisys['Time_ROI'][0] + 1):
                Tmp[ki] = 0

            CostFunNBS_PTAP = Tmp
            CostFunBaU_PTAP = Tmp

        CostFunNBS_Cap = CostFunNBS_PTAP * 0
        CostFunNBS_Cap['Process'] = 'Intake_NoData'
        CostFunNBS_Cap['Cost_Function'] = 0
        CostFunBaU_Cap = CostFunNBS_Cap

    # Calculo de Beneficio - Total
    Benefit_Cap     = CostFunBaU_Cap.values[:,2:].T - CostFunNBS_Cap.values[:,2:].T
    Benefit_PTAP    = CostFunBaU_PTAP.values[:,2:].T - CostFunNBS_PTAP.values[:,2:].T
    
    # Control para que los beneficios no sean negativos [Bug - 24-03-2023]
    Benefit_Cap[Benefit_Cap < 0] = 0
    Benefit_PTAP[Benefit_PTAP < 0] = 0
    
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
            Posi = np.arange(j,int(t_roi),int(CostNBS[3:].values[0][i]))
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
    DifCO2_1 = (C_NBS.sum(1) - C_BaU.sum(1)) # Se estima la diferencia de almacenamiento de CO2 entre los dos ecenarios
    DifCO2_2 = DifCO2_1.diff() # Se estima la diferencia de los diferenciales de alamcenamiento de CO2 por año
    DifCO2_2[np.isnan(DifCO2_2)] = 0 # Todos los valores NaN son iguales a cero
    DifCO2_2[DifCO2_2 < 0.001] = 0 # todos los valores por debajo de 0.001 se consideran cero
    Carbons = Factor*DifCO2_2*TD['Value'][5] # Se pasa de CO2 a dinero
    Carbons = Carbons.values[1:]

    # Total de costo de procesos + Carbono
    TotalBenefit_1      = np.sum(Benefit_Cap,1) + np.sum(Benefit_PTAP,1) + Carbons
    TotalBenefit_1_TD_1 = TotalBenefit_1/((1 + TD['Value'][2])**np.arange(1,t_roi + 1))
    TotalBenefit_1_TD_2 = TotalBenefit_1/((1 + TD['Value'][3])**np.arange(1,t_roi + 1))
    TotalBenefit_1_TD_3 = TotalBenefit_1/((1 + TD['Value'][4])**np.arange(1,t_roi + 1))

    # Total de costo de NBS
    TotalBenefit_2       = Cost_M.sum(1) + Cost_T + Cost_O.sum(1) + Cost_I.sum(1) + Cost_P.sum(1)
    TotalBenefit_2_TD_1  = TotalBenefit_2/((1 + TD['Value'][2])**np.arange(1,t_roi + 1))
    TotalBenefit_2_TD_2  = TotalBenefit_2/((1 + TD['Value'][3])**np.arange(1,t_roi + 1))
    TotalBenefit_2_TD_3  = TotalBenefit_2/((1 + TD['Value'][4])**np.arange(1,t_roi + 1))

    ROI_0 = TotalBenefit_1.sum()/TotalBenefit_2.sum()
    ROI_1 = TotalBenefit_1_TD_1.sum()/TotalBenefit_2_TD_1.sum()
    ROI_2 = TotalBenefit_1_TD_2.sum()/TotalBenefit_2_TD_2.sum()
    ROI_3 = TotalBenefit_1_TD_3.sum()/TotalBenefit_2_TD_3.sum()

    # NPV - I, M, O, T, P
    NPV_I = Cost_I.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    NPV_M = Cost_M.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    NPV_O = Cost_O.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    NPV_T = Cost_T / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    NPV_P = Cost_P.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))

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

    NPV = pd.DataFrame(data=np.zeros((1, 7)), columns=['Implementation', 'Maintenance', 'Oportunity', 'Transaction', 'Platform', 'Benefit', 'Total'])
    NPV['Implementation']   = -1*NPV_I.sum()
    NPV['Maintenance']      = -1*NPV_M.sum()
    NPV['Oportunity']       = -1*NPV_O.sum()
    NPV['Transaction']      = -1*NPV_T.sum()
    NPV['Platform']         = -1*NPV_P.sum()
    NPV['Benefit']          = TotalBenefit_1_TD_1.sum()
    NPV['Total']            = NPV.sum(1)

    Total_2 = Total_2.set_index(np.arange(0, t_roi) + 1)
    Total_3 = Total_3.set_index(np.arange(0, t_roi) + 1)

    Total_4_0 = pd.DataFrame(data=Cost_I,columns=NameNBS,index=NameIndex)
    Total_5_0 = pd.DataFrame(data=Cost_M,columns=NameNBS,index=NameIndex)
    Total_6_0 = pd.DataFrame(data=Cost_O,columns=NameNBS,index=NameIndex)
    Total_7_0 = pd.DataFrame(data=Cost_T,columns=['Cost'],index=NameIndex)
    Total_8_0 = pd.DataFrame(data=Cost_P,columns=TD['Cost'][5:],index=NameIndex)

    Total_9_0 = CostFunBaU_Cap.groupby(by='Process').sum() - CostFunNBS_Cap.groupby(by='Process').sum()
    del Total_9_0['Cost_Function']
    
    # Control para que los beneficios no sean negativos - [Bug - 24-03-2023]
    Total_9_0[Total_9_0<0] = 0

    Total_10_0 = CostFunBaU_PTAP.groupby(by='Process').sum() - CostFunNBS_PTAP.groupby(by='Process').sum()
    del Total_10_0['Cost_Function']
    
    # Control para que los beneficios no sean negativos - [Bug - 24-03-2023]
    Total_10_0[Total_10_0 < 0] = 0
    
    Total_11_0 = pd.DataFrame(data=Carbons,columns=['Carbons'],index=NameIndex)

    # TD min
    Total_4_1  = Total_4_0 / ((1 + (np.ones(Total_4_0.shape)*TD['Value'][3]))**np.tile(np.arange(1,t_roi + 1),(Total_4_0.shape[1],1)).transpose())
    Total_5_1  = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_1  = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_1  = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_1  = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_1  = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_9_0.shape[0], 1)))
    Total_10_1 = Total_10_0 / ((1 + (np.ones(Total_10_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_10_0.shape[0], 1)))
    Total_11_1 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    #C1 = Total_4_1.sum(1) + Total_5_1.sum(1) + Total_6_1.sum(1) + Total_7_1.sum(1) + Total_8_1.sum(1)
    #B1 = Total_9_1.sum().values + Total_10_1.sum().values + np.reshape(Total_11_1.values,(1,30))[0]

    # TD Mean
    Total_4_2 = Total_4_0 / ((1 + (np.ones(Total_4_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_4_0.shape[1], 1)).transpose())
    Total_5_2 = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_2 = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_2 = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_2 = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_2 = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_9_0.shape[0], 1)))
    Total_10_2 = Total_10_0 / ((1 + (np.ones(Total_10_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_10_0.shape[0], 1)))
    Total_11_2 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    #C2 = Total_4_2.sum(1) + Total_5_2.sum(1) + Total_6_2.sum(1) + Total_7_2.sum(1) + Total_8_2.sum(1)
    #B2 = Total_9_2.sum() + Total_10_2.sum() + np.reshape(Total_11_2.values,(1,30))[0]

    # TD max
    Total_4_3 = Total_4_0 / ((1 + (np.ones(Total_4_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_4_0.shape[1], 1)).transpose())
    Total_5_3 = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_3 = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_3 = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_3 = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_3 = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_9_0.shape[0], 1)))
    Total_10_3 = Total_10_0 / ((1 + (np.ones(Total_10_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_10_0.shape[0], 1)))
    Total_11_3 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    #C3 = Total_4_3.sum(1) + Total_5_3.sum(1) + Total_6_3.sum(1) + Total_7_3.sum(1) + Total_8_3.sum(1)
    #B3 = Total_9_3.sum() + Total_10_3.sum() + np.reshape(Total_11_3.values,(1,30))[0]

    Total_4_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'1.0_Implementation_Costs.csv'),index_label='Time')
    Total_5_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'2.0_Maintenance_Costs.csv'),index_label='Time')
    Total_6_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'3.0_Opportunity_Costs.csv'),index_label='Time')
    Total_7_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'4.0_Transaction_Costs.csv'),index_label='Time')
    Total_8_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'5.0_Platform_Costs.csv'),index_label='Time')
    Total_9_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'6.0_Cap_Saves.csv'))
    Total_10_0.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'7.0_PTAP_Saves.csv'))
    Total_11_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '8.0_Carbons_Saves.csv'))

    Total_4_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.1_Implementation_Costs.csv'), index_label='Time')
    Total_5_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.1_Maintenance_Costs.csv'), index_label='Time')
    Total_6_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.1_Opportunity_Costs.csv'), index_label='Time')
    Total_7_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.1_Transaction_Costs.csv'), index_label='Time')
    Total_8_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.1_Platform_Costs.csv'), index_label='Time')
    Total_9_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.1_Cap_Saves.csv'))
    Total_10_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.1_PTAP_Saves.csv'))
    Total_11_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '8.1_Carbons_Saves.csv'))

    Total_4_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.2_Implementation_Costs.csv'), index_label='Time')
    Total_5_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.2_Maintenance_Costs.csv'), index_label='Time')
    Total_6_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.2_Opportunity_Costs.csv'), index_label='Time')
    Total_7_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.2_Transaction_Costs.csv'), index_label='Time')
    Total_8_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.2_Platform_Costs.csv'), index_label='Time')
    Total_9_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.2_Cap_Saves.csv'))
    Total_10_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.2_PTAP_Saves.csv'))
    Total_11_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '8.2_Carbons_Saves.csv'))

    Total_4_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.3_Implementation_Costs.csv'), index_label='Time')
    Total_5_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.3_Maintenance_Costs.csv'), index_label='Time')
    Total_6_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.3_Opportunity_Costs.csv'), index_label='Time')
    Total_7_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.3_Transaction_Costs.csv'), index_label='Time')
    Total_8_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.3_Platform_Costs.csv'), index_label='Time')
    Total_9_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.3_Cap_Saves.csv'))
    Total_10_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.3_PTAP_Saves.csv'))
    Total_11_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '8.3_Carbons_Saves.csv'))

    Total_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '9_GlobalTotals.csv'), index_label='Time')
    Total_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '10_Benefit_Sensitivity.csv'), index_label='Time')
    Total_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '11_Cost_Sensitivity.csv'), index_label='Time')
    ROI.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '12_ROI_Sensitivity.csv'))
    NPV.to_csv(os.path.join(PathProject_ROI,OUTPUTS,'13_NPV.csv'))
