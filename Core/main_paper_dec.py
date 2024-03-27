# -*- coding: utf-8 -*-## @namespace main_paper
# Created on Tue March 26 2023
# Author
# Alejandro Pena-Bello and Marten van der Kam
# alejandro.penabello@hevs.ch; marten.vanderkam@unibas.ch
# Modification of main script used for the papers Optimized PV-coupled battery systems for combining applications: Impact of battery technology and geography (Pena-Bello et al 2019) and Decarbonizing heat with PV-coupled heat pumps supported by electricity and heat storage: Impacts and trade-offs for prosumers and the grid (Pena-Bello et al 2021)
# This enhancement includes the use electric vehicles together with the previously assessed PV, battery system and HP and thermal storage. We study the different applications which residential batteries can perform from a consumer perspective. Applications such as avoidance of PV curtailment, demand load-shifting and demand peak shaving are considered along  with the base application, PV self-consumption. It can be used with six different battery technologies currently available in the market are considered as well as three sizes (3 kWh, 7 kWh and 14 kWh). We analyze the impact of the type of demand profile and type of tariff structure by comparing results across dwellings in Switzerland. The electric vehicles chargers that can be chosen are 3.6, 7, or 11 kW.
# The HP, battery and TS schedule is optimized for every day (i.e. 24 h optimization framework), we assume perfect day-ahead forecast of the electric vehicle use, the electricity and heat demand load and solar PV generation in order to determine the maximum economic potential regardless of the forecast strategy used. Battery aging was treated as an exogenous parameter, calculated on daily basis and was not subject of optimization. Data with 15-minute temporal resolution were used for simulations. The model objective function have two components, the energy-based and the power-based component, as the tariff structure depends on the applications considered, a boolean parameter activate the power-based factor of the bill when is necessary.

# The script works in Linux and Windows
# This script works was tested with pyomo version 5.4.3
# INPUTS
# ------
# OUTPUTS
# -------
# TODO
# ----
# User Interface, including path to save the results and choose countries, load curves, etc.
# Simplify by merging select_data and load_data and probably load_param.
# Requirements
# ------------
#  Pandas, numpy, sys, glob, os, csv, pickle, functools, argparse, itertools, time, math, pyomo and multiprocessing

import sys
if sys.platform!='win32':
    import resource
    
import os
import pandas as pd
import argparse
import numpy as np
import itertools

from dateutil import parser
import glob
import multiprocessing as mp
import time
import paper_classes as pc
from functools import wraps
from pathlib import Path
import traceback
import pickle
import csv
from Core_LP import single_opt2
import post_proc as pp

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.__name__, str(t1-t0))
               )
        return result
    return function_timer

def expand_grid(dct): #MAIN PARAMETERS MULTIPLICATION TO CONSTRUCT SCENARIOS
    rows = itertools.product(*dct.values())
    return pd.DataFrame.from_records(rows, columns=dct.keys())

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_param(combinations):
    '''
    Description
    -----------
    Load all parameters into a dictionary, if aging is present (True) or not
    (False), percentage of curtailment, Inverter and Converter efficiency, time
    resolution (0.25), number of years or days if only some days want to be
    optimized, applications, capacities and technologies to optimize.

    Applications are defined as a Boolean vector, where a True activates the
    corresponding application. PVSC is assumed to be always used. The order
    is as follows: [PVCT, PVSC, DLS, DPS]
	i.e PV avoidance of curtailment, PV self-consumption,
	Demand load shifting and demand peak shaving.
    [0,1,0,0]-0
    [0,1,0,1]-1
    [0,1,1,0]-2
    [0,1,1,1]-3
    [1,1,0,0]-4
    [1,1,0,1]-5
    [1,1,1,0]-6
    [1,1,1,1]-7


    Parameters
    ----------
    PV_nominal_power : int

    Returns
    ------
    param: dict
    Comments
    -----
    week 18 day 120 is transtion from cooling to heating WHY THESE MONTHS
    week 40 day 274 is transtion from cooling to heating
    '''
    print('##############')
    print('load data')
    
    'FIRST, A HOUSE WITH A CORRESPONDING PROFILE IS CHOSES'
    if combinations['electricity_profile'] == 'Low':
            electricity_profiles = pd.read_csv('../Input/HHLow.csv')
    
    if combinations['electricity_profile'] == 'Medium':
            electricity_profiles = pd.read_csv('../Input/HHMedium.csv')
    
    if combinations['electricity_profile'] == 'High':
            electricity_profiles = pd.read_csv('../Input/HHHigh.csv')
    
    combinations['hh'] = electricity_profiles.iloc[combinations['profile_row_number']]
    
    
    id_dwell=str(int(combinations['hh']))
    print(id_dwell)
    [clusters,PV,App_comb_df,conf,house_types,hp_types,rad_types]=pp.get_table_inputs()

    PV_nom=PV[PV.PV==combinations['PV_nom']].PV.values[0]
    quartile=PV[(PV.PV==combinations['PV_nom'])&(PV.country==combinations['country'])].quartile.values[0]
    App_comb=[str2bool(i) for i in App_comb_df[App_comb_df.index==int(combinations['App_comb'])].App_comb.values[0].split(' ')]
#####################################################
    design_param=load_obj('../Input/dict_design_oct')
    aging=True
    Inverter_power=round(PV_nom/1.2,1)
    Curtailment=0.5
    Inverter_Efficiency=0.95
    Converter_Efficiency_HP=0.98
    dt=0.25
    Capacity_tariff=9.39*12/365
    nyears=1
    days=365
    testing=True
    cooling=False
    week=1
######################################################

    filename_el=Path('../Input/Electricity_demand_supply_2017.csv')
    filename_heat=Path('../Input/preprocessed_heat_demand_2_new_Oct.csv')
    filename_prices=Path('../Input/Prices_2017.csv')
   
  
    fields_el=['index',id_dwell,'E_PV']
    new_cols=['E_demand', 'E_PV']

    df_el = pd.read_csv(filename_el,engine='python',sep=',|;',index_col=0,
                        parse_dates=[0],infer_datetime_format=True, usecols=fields_el)
    
    if np.issubdtype(df_el.index.dtype, np.datetime64):
        df_el.index=df_el.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    else:
        df_el.index=pd.to_datetime(df_el.index,utc=True)
        df_el.index=df_el.index.tz_convert('Europe/Brussels')

    df_el.columns=new_cols

    if (combinations['house_type']=='SFH15')| (combinations['house_type']=='SFH45'):HEATING
        aux_name1='SFH15_45'
        aux_name2=combinations['house_type']
    else:
        aux_name1='SFH100'
        aux_name2='SFH100'

    fields_heat=['index','Set_T','Temp', aux_name2+'_kWh','DHW_kWh', 'Temp_supply_'+aux_name1,'Temp_supply_'+aux_name1+'_tank',
                'COP_'+aux_name2,'hp_'+aux_name2+'_el_cons','COP_'+aux_name2+'_DHW',
                 'hp_'+aux_name2+'_el_cons_DHW','COP_'+aux_name2+'_tank',
                 'hp_'+aux_name2+'_tank_el_cons']
    new_cols=['Set_T','Temp', 'Req_kWh','Req_kWh_DHW','Temp_supply','Temp_supply_tank','COP_SH','COP_tank','COP_DHW',
              'hp_sh_cons','hp_tank_cons','hp_dhw_cons']
    df_heat=pd.read_csv(filename_heat,engine='python',sep=';',index_col=[0],
                        parse_dates=[0],infer_datetime_format=True, usecols=fields_heat)
    df_heat.columns=new_cols


    if np.issubdtype(df_heat.index.dtype, np.datetime64):
        df_heat.index=df_heat.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    else:
        df_heat.index=pd.to_datetime(df_heat.index,utc=True)
        df_heat.index=df_heat.index.tz_convert('Europe/Brussels')

    fields_prices=['index', 'Price_flat', 'Price_DT', 'Export_price', 'Price_flat_mod',
   'Price_DT_mod']
    df_prices=pd.read_csv(filename_prices,engine='python',sep=',|;',index_col=[0],
                        parse_dates=[0],infer_datetime_format=True ,usecols=fields_prices)

    if np.issubdtype(df_prices.index.dtype, np.datetime64):
        df_prices.index=df_prices.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    else:
        df_prices.index=pd.to_datetime(df_prices.index,utc=True)
        df_prices.index=df_prices.index.tz_convert('Europe/Brussels')
        

    ########## LOAD EV DATA
    
    'WE FIRST EXTRACT THE CORRESPONDING ID NUMBERS'
    if combinations['EV_use'] == 'Low':
            EV_IDs = pd.read_csv('../Input/hhnrEVLow.csv')
    
    if combinations['EV_use'] == 'Medium':
            EV_IDs = pd.read_csv('../Input/hhnrEVMedium.csv')
    
    if combinations['EV_use'] == 'High':
            EV_IDs = pd.read_csv('../Input/hhnrEVHigh.csv')
            
    if combinations['EV_use'] == 'None': #It will select a number for the model to work, but it is not used later
            EV_IDs = pd.read_csv('../Input/hhnrEVHigh.csv')
    
    EV_ID = EV_IDs.iloc[combinations['profile_row_number']]['HHNR_WEEKDAY_WEEKENDAY']
    
    print(EV_ID)
        
    
    if ( combinations['EV_P_max'] != '3_6' and combinations['EV_P_max'] != '7' and combinations['EV_P_max'] != '11' ): print('Invalid charging power, choose 3_6, 7, or 11')
    
    
    
    filename_EV = Path('../Input/dfEVBasopra.csv')
    fields_EV=['index','energyRequired'+combinations['EV_P_max']+'kW'+ combinations['EV_use'],'maxPower'+ combinations['EV_use'],'energyTrip'+ combinations['EV_use']]
    
   
    df_EV = pd.read_csv(filename_EV,engine='python',sep=',|;',index_col=0,
                         parse_dates=[0],infer_datetime_format=True, usecols=fields_EV)

    
    df_EV.columns=['E_EV_req','EV_home','E_EV_trip']
    
    'SEPERATE EV FILES FOR THE DIFFERENT PROFILES'
    if combinations['EV_use'] == 'Low':
            filename_EV2 = Path('../Input/dfEVLow.csv')
    
    if combinations['EV_use'] == 'Medium':
            filename_EV2 = Path('../Input/dfEVMedium.csv')
    
    if combinations['EV_use'] == 'High':
            filename_EV2 = Path('../Input/dfEVHigh.csv')
            
    if combinations['EV_use'] == 'None':
            filename_EV2 = Path('../Input/dfEVNone.csv')
    
    if combinations['EV_use'] == 'None':
        aux_nameEV='1'
    else:
        aux_nameEV=str(combinations['profile_row_number']+1)
    fields_EV2=['energyRequired'+combinations['EV_P_max']+'kW_'+ aux_nameEV,'energyTrip_'+ aux_nameEV,'maxPower_'+ aux_nameEV]
    df_EV2 = pd.read_csv(filename_EV2, usecols=lambda x: x.strip().strip('"') in fields_EV2)
    df_EV2.columns=['E_EV_req','E_EV_trip','EV_home']
    
    df_EV['E_EV_req'] = df_EV2['E_EV_req'].values
    df_EV['EV_home']=df_EV2['EV_home'].values
    df_EV['E_EV_trip']=df_EV2['E_EV_trip'].values

    if np.issubdtype(df_EV.index.dtype, np.datetime64):
        df_EV.index=df_EV.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    else:
        df_EV.index=pd.to_datetime(df_EV.index,utc=True)
        df_EV.index=df_EV.index.tz_convert('Europe/Brussels')
            
    ####### EV variables
    EV_batt_cap=combinations['EV_batt_cap']
    if ( combinations['EV_P_max'] == '3_6' ):  EV_P_max=3.6/4
    if ( combinations['EV_P_max'] == '7' ):  EV_P_max=7/4
    if ( combinations['EV_P_max'] == '11' ):  EV_P_max=11/4

    EV_use=combinations['EV_use']
    EV_SOC_max=1.0
    EV_SOC_min=0.2
    EV_efficiency=0.9


    ############ data profiles through time
    
    data_input=pd.concat([df_el,df_heat,df_prices,df_EV],axis=1,copy=True,sort=False)
    #skip the first DHW data since cannot be produced simultaneously with SH
    data_input.loc[(data_input.index.hour<2),'Req_kWh_DHW']=0
    T_var=data_input.Temp.resample('1d').mean()
    data_input.loc[:,'E_PV']=data_input.loc[:,'E_PV']*PV_nom
    data_input['T_var']=T_var
    data_input.T_var=data_input.T_var.fillna(method='ffill')
    data_input['Cooling']=0
    data_input.loc[((data_input.index.month<=4)|(data_input.index.month>=10))&(data_input.Req_kWh<0),'Req_kWh']=0
    if cooling:
        data_input.loc[(data_input.index.month>4)&(data_input.index.month<10)&(data_input.T_var>20),'Cooling']=1#is T_var>20 then we need to cool only
        data_input.loc[(data_input.Cooling==1)&(data_input.Req_kWh>0),'Req_kWh']=0#if we should cool then ignore the heating requirements
        data_input.loc[(data_input.Cooling==1),'Req_kWh']=abs(data_input.loc[(data_input.Cooling==1),'Req_kWh'])

    data_input.loc[(data_input.index.month>4)&(data_input.index.month<10)&(data_input.Cooling==0),'Req_kWh']=0#if we should heat then ignore the cooling requirements
    data_input['Temp']=data_input['Temp']+273.15
    data_input['Set_T']=data_input['Set_T']+273.15
    data_input['Temp_supply']=data_input['Temp_supply']+273.15
    data_input['Temp_supply_tank']=data_input['Temp_supply_tank']+273.15
    data_input.loc[data_input.index.dayofyear==120,'Req_kWh']=0
    data_input.loc[data_input.index.dayofyear==274,'Req_kWh']=0
    data_input.loc[(data_input.index.dayofyear<120)|(data_input.index.dayofyear>274),'season']=0#'heating'
    data_input.loc[data_input.index.dayofyear==120,'season']=1#'transition_heating_cooling'
    data_input.loc[(data_input.index.dayofyear>120)&(data_input.index.dayofyear<274),'season']=2#'cooling'
    data_input.loc[data_input.index.dayofyear==274,'season']=3#'transition_cooling_heating'
    if data_input[((data_input.index.dayofyear<120)|(data_input.index.dayofyear>274))&(data_input.Temp_supply==data_input.Temp_supply_tank)].empty==False:
        data_input.loc[((data_input.index.dayofyear<120)|(data_input.index.dayofyear>274))&(data_input.Temp_supply==data_input.Temp_supply_tank),'Temp_supply_tank']+=1.5

    if testing:
        data_input=data_input[data_input.index.week==week]
        nyears=1
        days=7
        ndays=7

    print('##############')
    print('load parameters')
    conf=combinations['conf']
    print('conf')
    print(conf)

    #configuration=[Batt,HP,TS,DHW]
    #if all false, only PV
    conf_aux=[False,True,False,False]#[Batt,HP,TS,DHW]
    
    # For some settings the heat pump is removed
    if combinations['house_type']=='NoHeatPump':
        conf_aux[1]=False

    if conf<4:#No battery
        #print('inside batt')
        Converter_Efficiency_Batt=1
    else:
        conf_aux[0]=True
        Converter_Efficiency_Batt=0.98

    if (conf!=0)&(conf!=1)&(conf!=4)&(conf!=5)&(conf!=8)&(conf!=9):#TS present
        #print('inside TS')
        conf_aux[2]=True
        if (combinations['house_type']=='SFH15')|(combinations['house_type']=='SFH45'):
            tank_sh=pc.heat_storage_tank(mass=1500,surface=6)# For a 9600 liter tank with 3 m height and 2 m diameter 
            T_min_cooling=285.15#12°C
        else:
            tank_sh=pc.heat_storage_tank(mass=1500,surface=6) # For a 1500 liter tank with 1.426 m height and 1.25 diameter 
            T_min_cooling=285.15#12°C
    else:#No TS
        tank_sh=pc.heat_storage_tank(mass=0, surface=0.41)# For a 50 liter tank with .26 m height and .25 diameter WHY MASS IS 0 FOR 50 LITER TANK?
        T_min_cooling=0

    if (conf==1)|(conf==3)|(conf==5)|(conf==7):#DHW present
        #print('inside DHW')
        conf_aux[3]=True
        tank_dhw=pc.heat_storage_tank(mass=200, t_max=60+273.15, t_min=40+273.15,surface=1.6564) # For a 200 liter tank with 0.95 m height and .555 diameter
        if (conf==1)|(conf==5):
            if combinations['house_type']=='SFH15':
                tank_sh=pc.heat_storage_tank(mass=100, surface=1.209)# For a 100 liter tank with  .52m height and .25 diameter
            elif combinations['house_type']=='SFH45':
                tank_sh=pc.heat_storage_tank(mass=0, surface=1.913)# For a 200 liter tank with .52 m height and .35 diameter
            elif combinations['house_type']=='SFH100':
                tank_sh=pc.heat_storage_tank(mass=0, surface=3.2)# For a 50 liter tank with .52 m height and .5 diameter
    else:#No DHW
        tank_dhw=pc.heat_storage_tank(mass=1, t_max=0, t_min=0,specific_heat_dhw=0,U_value_dhw=0,surface_dhw=0)#null

    ndays=days*nyears
    #print(data_input.head())
    if combinations['HP']=='AS':
        if combinations['house_type']=='SFH15':
            Backup_heater=design_param['bu_15']+2
            hp=pc.HP_tech(technology='ASHP',power=design_param['hp_15'])
        elif combinations['house_type']=='SFH45':
            Backup_heater=design_param['bu_45']+4
            hp=pc.HP_tech(technology='ASHP',power=design_param['hp_45'])
        else:
            Backup_heater=design_param['bu_100']+17            
            hp=pc.HP_tech(technology='ASHP',power=design_param['hp_100'])
    elif combinations['HP']=='GSHP':
        #TODO
        pass 
    
    
    param={'conf':conf_aux,
    'aging':aging,'Inv_power':Inverter_power,
    'Curtailment':Curtailment,'Inverter_eff':Inverter_Efficiency,
    'Converter_Efficiency_HP':Converter_Efficiency_HP,
    'Converter_Efficiency_Batt':Converter_Efficiency_Batt,
    'delta_t':dt,'nyears':nyears,'T_min_cooling':T_min_cooling,
    'days':days,'ndays':ndays,'hp':hp,'tank_dhw':tank_dhw,'tank_sh':tank_sh,
    'Backup_heater':Backup_heater,'Capacity':combinations['Capacity'],'Tech':combinations['Tech'],
    'App_comb':App_comb,'cases':combinations['cases'],'ht':combinations['house_type'],
    'HP_type':combinations['HP'],'testing':testing, 'Cooling_ind':cooling,'name':str(id_dwell)+'_'+combinations['country']+'_PV'+str(quartile),
    'PV_nom':PV_nom,'Capacity_tariff':Capacity_tariff,
    'EV_batt_cap':EV_batt_cap,'EV_P_max':EV_P_max,'EV_ID':EV_ID,'EV_use':EV_use,'EV_SOC_max':EV_SOC_max,'EV_SOC_min':EV_SOC_min,
    'EV_efficiency':EV_efficiency,'electricity_profile':combinations['electricity_profile'],'profile_row_number':combinations['profile_row_number']}
    return param,data_input

def pooling2(combinations):
    '''
    Description
    -----------
    Calls other functions, load the data and Core_LP.
    This function includes the variables for the tests (testing and data_input.index.week)
    Parameters
    ----------
    selected_dwellings : dict

    Returns
    ------
    bool
        True if successful, False otherwise.


    '''

    print('##########################################')
    print('pooling')
    print(combinations)
    print('##########################################')
    param,data_input=load_param(combinations)
    try:
        if param['nyears']>1:
            data_input=pd.DataFrame(pd.np.tile(pd.np.array(data_input).T,
                                   param['nyears']).T,columns=data_input.columns)
        print('#############pool################')
        #print(param['tank_dhw'].mass)
        #print(param['tank_sh'].mass)
        [df,aux_dict]=single_opt2(param,data_input)
        print('out of optimization')
    except IOError as e:
        print ("I/O error({0}): {1}".format(e.errno, e.strerror))
        raise

    except :
        print ("Back to main.")
        raise
    return


@fn_timer
def main():
    '''
    Main function of the main script. Allows the user to select the country
	(CH or US). For the moment is done automatically if working in windows
	the country is US. It opens a pool of 4 processes to process in parallel, if several dwellings are assessed.
    '''
    print(os.getcwd())
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('Welcome to basopra')
    print('Here you will able to optimize the schedule of PV-coupled HP systems and an electric vehicle with electric and/or thermal storage for a given electricity demand')

    try:
        filename=Path('../Output/aggregated_results.csv')
        df_done=pd.read_csv(filename,sep=';|,',engine='python',index_col=False).drop_duplicates()
        #df_done.drop(df_done.tail(2).index,inplace=True)
        aux=df_done.groupby([df_done.Capacity,df_done.App_comb,df_done.Tech,df_done.PV_nom,df_done.hh,df_done.country,df_done.cases,df_done.conf,df_done.HP,df_done.house_type,df_done.EV_batt_cap,df_done.EV_P_max,df_done.EV_use]).size().reset_index()
       
    except IOError:
        #There is not such a file, then create it
        cols=['Bool_char','Bool_cons','Bool_dis','Bool_hp','Bool_hpdhw','Bool_inj','E_EV','E_EV_charge','E_EV_discharge','E_EV_loss','E_PV_EV','E_PV_batt','E_PV_bu','E_PV_budhw','E_PV_curt','E_PV_grid','E_PV_hp','E_PV_hpdhw','E_PV_load','E_batt_EV','E_batt_bu','E_batt_budhw','E_batt_hp','E_batt_hpdhw','E_batt_load','E_bu','E_budhw','E_char','E_cons','E_cons_without_backup','E_dis','E_grid_EV','E_grid_batt','E_grid_bu','E_grid_budhw','E_grid_hp','E_grid_hpdhw','E_grid_load','E_hp','E_hpdhw','E_loss_Batt','E_loss_conv','E_loss_inv','E_loss_inv_PV','E_loss_inv_batt','E_loss_inv_grid','Q_dhwst_hd','Q_hp_sh','Q_hp_ts','Q_loss_dhwst','Q_loss_ts','Q_ts','Q_ts_delta','Q_ts_sh','T_dhwst','T_ts','E_demand','E_PV','Req_kWh','Req_kWh_DHW','Set_T','Temp','Temp_supply','Temp_supply_tank','T_aux_supply','COP_tank','COP_SH','COP_DHW','Cooling','E_demand_hp_pv_dhw','E_demand_hp_pv','E_demand_pv','E_demand_hp_pv_dhw_EV','E_demand_hp_pv_EV','E_demand_pv_EV','TSC','DSC','ISC','CU','TSS','App_comb','conf','Capacity','Tech','PV_nom','profile_row_number','cluster','electricity_profile','hh','country','quartile','cases','house_type','HP','EV_batt_cap','EV_P_max','EV_ID','EV_use','SOC_mean','P_drained_max','P_drained_without_backup_max','P_injected_max','last_cap','cap_fading','last_SOH','cycle_to_total'] 
        with open(filename, "w+") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(cols)

        aux=pd.DataFrame()
    finally:
        #Define the different combinations of inputs to be run
        dct={'Capacity':[7],'App_comb':[0],'Tech':['NMC',],'PV_nom':[4.8],'country':['CH'],'cases':['mean'],'conf':[0,4],'house_type':['SFH15','SFH45','SFH100','NoHeatPump'],'HP':['AS'],'electricity_profile':['Low','Medium','High'],'hh':[""],'EV_batt_cap':[60],'EV_P_max':['7'],'EV_use':['High','Medium','Low','None'],'profile_row_number':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]}

        Total_combs=expand_grid(dct)
        #print(Total_combs.dtypes)
        #print(aux)
        if aux.empty:
            Combs_todo=Total_combs.copy()
        else:

            Combs_todo=aux.merge(Total_combs,how='outer',indicator=True)#Warning

            Combs_todo=Combs_todo[Combs_todo['_merge']=='right_only']
            Combs_todo=Combs_todo.loc[:,Combs_todo.columns[:-1]]

        print(Combs_todo.head())
        Combs_todo=[dict(Combs_todo.loc[i,:]) for i in Combs_todo.index]
        print(len(Combs_todo))
        mp.freeze_support()
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        pool=mp.Pool(processes=25)
        #selected_dwellings=select_data(Combs_todo)
        #print(selected_dwellings)
        #print(Combs_todo)
        pool.map(pooling2,Combs_todo)
        pool.close()
        pool.join()
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))
    print(get_memory() * 1024 / 2)

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemTotal:'):
                free_memory += int(sline[1])

    return free_memory
if __name__ == '__main__':
    if sys.platform!='win32':
        memory_limit() # Limitates maximun memory usage to half
        try:
            main()
        except MemoryError:
            sys.stderr.write('MAXIMUM MEMORY EXCEEDED')
            sys.exit(-1)
    else:
        main()
