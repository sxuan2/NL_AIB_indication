# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 01:34:12 2020

@author: sijian.xuan
"""


import pandas as pd
import numpy as np
from time import time
from pandasql import sqldf
from functools import reduce


def read_file(UNIV="Q:/DATA/XTREND/Univ17",
              Geosp = "X:/Data/Geospatial_CR_Herprojectie"):
    """
    now this location only has SAS file,
    so to make this work the corresponding csv file need to be created 
    now we will use our location instead
    """
    brick_prov = pd.read_csv(UNIV + "/pc4_brick_2017.csv", engine ='python')
    brick_prov = brick_prov[["brick","PROVnaam"]]
    brick_prov.drop_duplicates(inplace = True)
    brick_prov.sort_values(by = ["brick","PROVnaam"], inplace=True)

    phdb_x_y = pd.read_csv(Geosp + "/apotheken_universum_20170215.csv", engine ='python')
    phdb_x_y = phdb_x_y[["phdb","x","y"]]
    phdb_x_y.drop_duplicates(inplace = True)
    phdb_x_y = phdb_x_y[phdb_x_y.phdb>0]
    phdb_x_y.sort_values(by = "phdb", inplace=True)
    return brick_prov, phdb_x_y

# a = 'C:/Users/sijian.xuan/Desktop/python_version/input'
a = "C:/Users/sijian.xuan/Desktop/NL AIB pythontransfer/python_version/input"
brick_prov, phdb_x_y = read_file(UNIV=a,Geosp=a)

# brick_prov, phdb_x_y = read_file()

def pysqldf(q):
    start = time()
    output = sqldf(q, globals())
    end = time()
    print("SQL running time: {} s".format(end-start))
    print("output file shape: {}".format(output.shape))
    return(output)

def _age_gender_info(input_file):
    print("calculating age gender information...")
    Focus_age_gender = input_file[['pat', 'sex', 'age','numdax']]
    # Focus_age_gender['numdax'] = pd.DataFrame(pd.to_datetime(Focus_age_gender['numdax'], format = '%d/%m/%Y'))
    Focus_age_gender.drop_duplicates(inplace = True)
    Focus_age_gender.sort_values(by = ['pat', 'numdax'], ascending = [True, False], inplace=True)
    Focus_age_gender.drop_duplicates(subset = ['pat'], inplace = True)
    Features_age_gender = Focus_age_gender[['pat','age','sex']]
    Features_age_gender = Features_age_gender.rename(index=str, columns={'sex':'gender'})
    print("ENDING EXECUTION: _age_gender_info")
    return (Features_age_gender)

def _market_consumption(input_file, label, pack_id):
    print("calculating market consumption information...")
    global input_file1
    input_file1 = input_file
    q = """
        select *,
            (Focus_Euro/duration)*365 as yearly_euro,
            (Focus_cu/duration)*365 as yearly_cu,
            (Focus_Rx*1.0/duration)*365 as yearly_rx
        from
            (select *, (lastdax - firstdax + 1) as duration
            from
                (       
                select pat,
                    min(numdax) as firstdax, 
                    max(numdax + td - 1) as lastdax,
                    count(*) as Focus_Rx, 
                    sum(euro) as Focus_Euro, 
                    sum(cu) as Focus_cu,
                    count(distinct atc4) as nr_atc4,
                    count(distinct line) as nr_line,
                    count(distinct """ + pack_id +""") as nr_pack_info,
                    count(distinct phdb) as nr_phdb
                from input_file1
        group by pat))
    """        
    
    Features_focus_cons_1  = sqldf(q, globals())
    # rename columns
    col_list = Features_focus_cons_1.columns.tolist()
    for i in range(len(col_list)):
        if col_list[i] == "pat":
            pass
        elif col_list[i][:2] == "nr":
            col_list[i] = "nr_" + label + col_list[i][2:]
        elif col_list[i][:5] == "Focus":
            col_list[i] = label + col_list[i][5:]
        else: col_list[i] = label + "_" + col_list[i]
        
        if "pack_info" in col_list[i]:
            col_list[i] = col_list[i].replace("pack", pack_id)
            
    Features_focus_cons_1.columns = col_list
    print("ENDING EXECUTION: _market_consumption")
    return(Features_focus_cons_1)

def _market_compliance(input_file, label):
    print("calculating compliance information...")
    global input_file1
    input_file1 = input_file
    q = """
        select pat, numdax as fromdax , td, (numdax + td - 1) as todax
        from
        (select pat,
        numdax,
        max(td) as td
        from 
        input_file1
        group by pat,numdax
        order by pat,numdax)
        """    
    compliance_raw_1 = pysqldf(q)
    compliance_raw_1.sort_values(by = ["pat","fromdax"],inplace = True)  
    m = 0
    n = 1
    while(m!=n):
        cond111, = np.where((compliance_raw_1.pat == compliance_raw_1.pat.shift())&(compliance_raw_1.todax <= compliance_raw_1.todax.shift()))
        cond111 = cond111.tolist()
        m = compliance_raw_1.shape[0]
        compliance_raw_1 = compliance_raw_1.drop(cond111)
        n = compliance_raw_1.shape[0]
        compliance_raw_1.reset_index(drop = True, inplace=True)
    Compliance_raw_02 = compliance_raw_1.copy()
    Compliance_raw_02['first'] =  np.where(Compliance_raw_02.pat != Compliance_raw_02.pat.shift(),1,0)
    global Compliance_raw_03
    Compliance_raw_03 = Compliance_raw_02[['pat','fromdax','td','todax','first']]
    Compliance_raw_03['prevdax'] = Compliance_raw_03.todax.shift()
    cond, = np.where(Compliance_raw_03['first'] == 1)
    cond = cond.tolist()
    Compliance_raw_03['prevdax'][cond] = np.nan
    cond1, = np.where(Compliance_raw_03['first'] != 1)
    cond1 = cond1.tolist()
    Compliance_raw_03['gap'] = 0
    Compliance_raw_03['gap'][cond1] =Compliance_raw_03['fromdax'][cond1] - Compliance_raw_03['prevdax'][cond1]
    cond2, = np.where(Compliance_raw_03['gap'] <= 0)
    cond2 = cond2.tolist()
    Compliance_raw_03['gap'][cond2] = 0
    Compliance_raw_03 = Compliance_raw_03.drop(['first'], axis=1)
    q = """
        select pat, duration , gap, max(0,((duration-gap)/duration)) as compliance_rate
        from
            (select pat,
                    (max(todax) - min(fromdax) + 1) as duration,
                    sum(gap) as gap
            from 
                Compliance_raw_03
                group by pat)
        """    
    Compliance_raw_04 = pysqldf(q)
    Compliance_raw_04 = Compliance_raw_04.fillna(0)
    Compliance_raw_04 = Compliance_raw_04[["pat","compliance_rate"]]
    Compliance_raw_04.columns = ["pat",label + "_compliance_rate"]
    print("ENDING EXECUTION: _market_compliance")
    return(Compliance_raw_04)

def _line_consumption(in_FCC, line, label):
    print("calculating "+ line +" consumption information...")
    global input_file1
    input_file1 = in_FCC

    q =""" select pat, """ + line +""", count(*) as rx from input_file1 group by pat, """ + line 
    focus_line = pysqldf(q)
    focus_line[line].value_counts()
    focus_line2 = focus_line.pivot_table(index = 'pat', columns = line, aggfunc = sum, fill_value = 0 )
    focus_line2.reset_index(inplace=True)
    focus_line2.columns = label + "_" + focus_line2.columns.get_level_values(1).map(str)
    focus_line2 = focus_line2.rename(columns={label + "_": "pat"})
    print("ENDING EXECUTION: _line_consumption")
    return focus_line2

def _spec_consumption(in_spec, label, TOPN, removeGP4FL="Y", specc_label="specc"):
    print("calculating specialty information...")
    global input_file1
    input_file1 = in_spec.copy()
    q = "select " + specc_label + ", count(*) as n from input_file1 where " +\
        specc_label + """ not in ("", "0") group by """ + specc_label + " order by n desc"
    globals()[specc_label + "_list"] = pysqldf(q)
    globals()[specc_label + "_list2"] = globals()[specc_label + "_list"].head(TOPN)
    globals()[label + "_" + specc_label + "_01"] = pd.merge(globals()[specc_label + "_list2"], in_spec, 
                                                how = "inner", on = specc_label)
    globals()[label + "_" + specc_label + "_02"] = _line_consumption(in_FCC = globals()[label + "_" + specc_label + "_01"],
                                                                    line = specc_label, label = label)
    
    input_file1 = in_spec.copy()
    
    if removeGP4FL == "Y":
        q = "select pat, numdax, " + specc_label + ", count(*) as rx from input_file1 where instr(" + specc_label + ""","Huisarts")=0
        group by pat, numdax, """ + specc_label
        globals()[specc_label + "_numdax_01"] = pysqldf(q)
    else:
        q = "select pat, numdax, " + specc_label + """, count(*) as rx from input_file1
        group by pat, numdax, """ + specc_label
        globals()[specc_label + "_numdax_01"] = pysqldf(q)

    def firstlast(infl, label, operation):
        print("calculating first/last pickup information...")
        q = "select * from " + specc_label + "_numdax_01 group by pat having numdax=" + operation +"(numdax)"        
        globals()[specc_label + "_numdax_02"] = pysqldf(q)
        
        q = "select a.pat, a." + specc_label + " from " + specc_label + "_numdax_02 a inner join " + specc_label + "_list2" + " b on a." + specc_label + "=b." + specc_label + " order by a.pat, a." + specc_label               
        globals()[specc_label + "_numdax_03"] = pysqldf(q)
        
        globals()[specc_label + "_numdax_03"] = globals()[specc_label + "_numdax_03"].drop_duplicates(subset = "pat")
        globals()[specc_label+ "_" + label] = _line_consumption(globals()[specc_label + "_numdax_03"], line = specc_label, label = label)
        
        print("ENDING EXECUTION: firstlast")
        
        return(globals()[specc_label+ "_" + label])
    
    firstlast(infl = globals()[specc_label + "_numdax_01"], label = label + "_Last", operation = "max")
    firstlast(infl = globals()[specc_label + "_numdax_01"], label = label + "_First", operation = "min")

    mergelist = [globals()[label+ "_" + specc_label + "_02"],
                globals()[specc_label+ "_" + label + "_Last"],
                globals()[specc_label+ "_" + label + "_First"]]
        
    out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
    print("ENDING EXECUTION: _specc_consumption")
    return(out_Feature)

def _seasonal_consumption(in_seasonal, label):
    print("calculating seasonal information...")
    Focus_seasonal_01 = in_seasonal[["pat","numdax"]]
    Focus_seasonal_01["numdax1"] = pd.to_datetime(Focus_seasonal_01["numdax"],unit='D',origin=pd.Timestamp('1960-01-01'))
    Focus_seasonal_01["mnd"] = Focus_seasonal_01["numdax1"].dt.month
    Focus_seasonal_01["quarter"] = (np.floor((Focus_seasonal_01["mnd"]-0.1)/3) + 1).astype(int)
    Focus_seasonal_02 = _line_consumption(in_FCC = Focus_seasonal_01, line = "mnd", label = label + "M")
    Focus_seasonal_03 = _line_consumption(in_FCC = Focus_seasonal_01, line = "quarter", label = label + "Q")
    mergelist = [Focus_seasonal_02, Focus_seasonal_03]   
    out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
    print("ENDING EXECUTION: _seasonal_consumption")
    return(out_Feature)

def _location_consumption(in_location, brick_prov, phdb_x_y, label, testmnd):
    print("calculating location information...")
    global test01
    global test02
    global Focus_location_05
    global Focus_location_06
    test01 = pd.read_csv("Q:/DATA/XTREND/Recr/test" + testmnd + ".csv")
    test01 = test01[['PeriodId', 'PharmacyId', 'RepcallId', 'PhdbId', 'ONTEST', 'Brick','Region']]
    test01.columns = ["maand", "phaid", "repcall", "phdb", "ontest", "brick", "mingeb"]
    test01.drop_duplicates(subset = ["phdb", "maand"], inplace=True)
    test01.sort_values(by = ["phdb","maand"], inplace=True)
    q = """    
    select a.phdb, a.brick
    from test01 a
    inner join
        (select phdb, max(maand) as maand
        from test01
        group by phdb) b
    on a.phdb = b.phdb
    and a.maand = b.maand    
    """
    test02 = pysqldf(q)
    q = "select a.phdb, b.* from test02 a inner join brick_prov b on a.brick = b.brick"        
    test03 = pysqldf(q)
    test03.drop_duplicates(subset = "phdb", inplace=True)
    test03.sort_values(by = "phdb", inplace=True)
    Focus_location_02 = pd.merge(test03, in_location,how = "inner", on = "phdb")
    Focus_location_03_brick = _line_consumption(Focus_location_02,"brick",label + "Brick")
    Focus_location_03_PROVnaam = _line_consumption(Focus_location_02,"PROVnaam",label + "P")
    Focus_location_04 = pd.merge(Focus_location_03_brick,Focus_location_03_PROVnaam,on="pat")
    Focus_location_02.sort_values(by = "phdb", inplace=True)
    Focus_location_05 = pd.merge(Focus_location_02,phdb_x_y,how="left",on="phdb")
    Focus_location_05["rx"] = 1
    q = """
        select brick, sum(x*rx)/sum(rx) as x_mean, sum(y*rx)/sum(rx) as y_mean
        from Focus_location_05
        where x > 0 and y > 0
        group by brick
        """
    brick_x_y = pysqldf(q)
    brick_x_y.sort_values(by = "brick", inplace = True)
    Focus_location_06 = pd.merge(Focus_location_05,brick_x_y,how="left",on="brick")
    Focus_location_06.loc[Focus_location_06["x"] > 0, "x"] = Focus_location_06["x_mean"]
    Focus_location_06.loc[Focus_location_06["y"] > 0, "y"] = Focus_location_06["y_mean"]
    Focus_location_06.drop(["phdb","x_mean","y_mean"], axis=1, inplace=True)
    q = """
        select pat, round(sum(x*rx)/sum(rx),0.001) as x_coordinate, 
                    round(sum(y*rx)/sum(rx),0.001) as y_coordinate
        from Focus_location_06
        group by pat
        """
    Focus_location_07 = pysqldf(q)
    out_location = pd.merge(Focus_location_04,Focus_location_07,on="pat")
    out_location.fillna(0, inplace=True)
    print("ENDING EXECUTION: _location_consumption")
    return out_location

def Generic_Feature_Selection(Focus = None, Comed= None, testmnd = None,
                              pack_id = 'fcc',Focus_Consumption='Y', 
                              Focus_Line='Y', Focus_Specc='Y',
                              TOPN=10, Focus_Seasonal='Y', Focus_Regional='Y',
                              Focus_Insurance='Y', Comed_Consumption='Y', 
                              Comed_Specc='Y', Comed_Line='N', 
                              Selected_Comed_list='N',cohort_size=300000):
    """
    Generic feature selection is the macro to produce features for indication split project
    """
    print("MACRO: Generic_Feature_Selection starting...")
    if Focus is not None:
        ## Demographic information - Latest age and gender - Must Have
        Features_age_gender = _age_gender_info(input_file = Focus)
        out_Feature = Features_age_gender.copy()
        print("{} features created...".format(out_Feature.shape[1]))
        # Total and annual consumption of focus market
        if Focus_Consumption == 'Y':
            Features_FMC = _market_consumption(input_file=Focus, label= "focus", pack_id=pack_id)
            Compliance_FMC =  _market_compliance(input_file=Focus, label= "focus")
            mergelist = [out_Feature,Features_FMC,Compliance_FMC]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Consumption needed, skipping Focus_Consumption...")
        ##Total consumption of focus market per line;
        if Focus_Line=='Y':
            Features_focus_line = _line_consumption(in_FCC = Focus, line="line", label="FL")
            mergelist = [out_Feature,Features_focus_line]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'], how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Line needed, skipping Focus_Line...")
        if Focus_Specc == "Y":
            Features_focus_specc = _spec_consumption(in_spec = Focus, label = "FS", 
                                                     TOPN=TOPN,removeGP4FL="Y", specc_label="specc")
            mergelist = [out_Feature,Features_focus_specc]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Specc needed, skipping Focus_Specc...")
        if Focus_Seasonal == "Y":
            Features_focus_seasonal = _seasonal_consumption(in_seasonal = Focus, label = "Focus")
            mergelist = [out_Feature,Features_focus_seasonal]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Seasonal needed, skipping Focus_Seasonal...")
        if Focus_Regional == "Y":
            Features_Focus_location = _location_consumption(in_location = Focus, 
                                                            brick_prov=brick_prov, 
                                                            phdb_x_y=phdb_x_y, 
                                                            label = "Focus", testmnd=testmnd)
            mergelist = [out_Feature,Features_Focus_location]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Regional needed, skipping Focus_Regional...")
        if Focus_Insurance == 'Y':
            Features_uzovi = _spec_consumption(in_spec = Focus, label = "FS", specc_label="uzovi",TOPN = TOPN)
            mergelist = [out_Feature,Features_uzovi]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            print("{} features created...".format(out_Feature.shape[1]))
        else:
            print("No Focus_Insurance needed, skipping Focus_Insurance...")
    else:
        raise ValueError("No Focus dataset!!")
        
    # comedication market
    if Comed is not None:
        if Comed_Consumption == 'Y':
            pat_list = pd.DataFrame(Comed.pat.unique()).reset_index()
            pat_list.columns = ["index","pat"]
            pat_list["cohort"] = np.floor(pat_list["index"].div(cohort_size))+1
        
            input_cohort_total = pd.merge(left=pat_list[["pat","cohort"]],right=Comed,how="right",on="pat")
            
            cohort_max = int(pat_list["cohort"].max())
            cohort_name_ls = []
            for i in range(cohort_max):
                print("{} cohorts in total, now processing cohort {}...".format(cohort_max,i+1))
                globals()["input_cohort_" + str(i+1)] = input_cohort_total[input_cohort_total["cohort"] == i+1]
                globals()["Comed_cohort_" + str(i+1)] = _market_consumption(input_file = globals()["input_cohort_" + str(i+1)],
                                                                            label = "Comed", pack_id = pack_id)
                cohort_name_ls.append(globals()["Comed_cohort_" + str(i+1)])

            Features_Comed_cons = pd.concat(cohort_name_ls)
            Features_Comed_cons.sort_values(by="pat",inplace=True)
            mergelist = [out_Feature,Features_Comed_cons]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
        if Comed_Line == 'Y':
            pass
            # maybe write this in the future
            # AIB indication split does not use this feature
        
        if Comed_Specc == 'Y':
            Features_comed_specc = _spec_consumption(in_spec=Comed, label="CS",TOPN=TOPN,removeGP4FL="Y", specc_label="specc")
            mergelist = [out_Feature,Features_comed_specc]
            out_Feature = reduce(lambda left,right: pd.merge(left,right,on=['pat'],how='left'), mergelist).fillna(0)
            
    print("MACRO: Generic_Feature_Selection ending...")
    return out_Feature


def vc_good(df, label="a", digit=2):
  if isinstance(df, pd.DataFrame):
    c = df[label].value_counts(dropna=False)
    p = df[label].value_counts(dropna=False, normalize=True).mul(100).round(digit).astype(str) + '%'
    print(pd.concat([c,p], axis=1, keys=['counts', '%']))
  elif isinstance(df, pd.Series):
    c = df.value_counts(dropna=False)
    p = df.value_counts(dropna=False, normalize=True).mul(100).round(digit).astype(str) + '%'
    print(pd.concat([c,p], axis=1, keys=['counts', '%']))
  else:
    raise ValueError("check input type!")
    
def get_labels(model_nr, inpatindi):
    print("MACRO: get_labels starting...")
    inpatindi = inpatindi[["pat","indication_M" + str(model_nr)]]
    inpatindi.dropna(inplace=True)
    inpatindi.columns = ["pat","indication"]
    vc_good(df=inpatindi,label="indication")
    print("MACRO: get_labels ending...")
    return inpatindi

def add_comed_features_per_model(in_comed,focus_indication,in_feature,model_nr=1,pack_id_type="fcc",knmp_univ=None):
    print("MACRO: add_comed_features_per_model starting...")
    global selected_line4model
    global in_comed1
    in_comed1 = in_comed.copy()
    selected_line4model = Selecting_Comedications(Comed=in_comed1,pack_id=pack_id_type,
                                                  Label_File=focus_indication, Label="indication",
                                                  knmp_univ=knmp_univ)
    print("selected_line4model shape {}".format(selected_line4model.shape))
    selected_line4model.sort_values(by=pack_id_type,inplace=True)
    selected_line4model.drop_duplicates(subset=[pack_id_type], inplace=True)
    
    selected_line4model = selected_line4model[pack_id_type]
    in_comed1 = in_comed1[["pat", pack_id_type, "line"]]
    
    q = "select a.*, b.* from selected_line4model a inner join in_comed1 b on a." + pack_id_type + "=b." + pack_id_type
    COMED_Explore_01 = pysqldf(q)
    
    globals()["Features_Comed_Line_M" + str(model_nr)]= _line_consumption(in_FCC=COMED_Explore_01, line="line", label="CM")
    in_feature.sort_values(by="pat", inplace=True)
    globals()["Features_Comed_Line_M" + str(model_nr)].sort_values(by="pat", inplace=True)
    out_Feature = pd.merge(left=in_feature,right=globals()["Features_Comed_Line_M" + str(model_nr)],on=['pat'],how='left')
    out_Feature.fillna(0, inplace=True)
    print("MACRO: add_comed_features_per_model ending...")
    return out_Feature

def Selecting_Comedications(Comed, pack_id, Label_File, Label, knmp_univ=None):
    print("MACRO: Selecting_Comedications starting...")
    global Label_File1
    global comed1
    global COMED_Selection_01
    global COMED_Selection_02
    global COMED_Selection_03
    global COMED_Selection_tot2
    global knmp_univ1
    global COMED_Selection_4model
    comed1 = Comed.copy()
    Label_File1 = Label_File.copy()
    knmp_univ1 = knmp_univ.copy()
    
    Label_File1.drop_duplicates(subset = ["pat"], inplace=True)
    Label_File1.sort_values(by = "pat", inplace =True)
    
    q = "select a.pat, " + pack_id + ", b." + Label + ", count(*) as rx_unproj from comed1 a inner join Label_File1 b on a.pat=b.pat group by a.pat,"  + pack_id + ", b." + Label
    COMED_Selection_01 = pysqldf(q)
    
    q = "select " + pack_id + ", " + Label + ", sum(rx_unproj) as rx_unproj, count(distinct pat) as pat_unproj from COMED_Selection_01 group by " + pack_id + "," + Label
    COMED_Selection_02 = pysqldf(q)

    q = "select " + Label + ", sum(rx_unproj) as rx_unproj, count(distinct pat) as pat_unproj from COMED_Selection_01 group by " + Label
    COMED_Selection_tot = pysqldf(q)

    COMED_Selection_03 = COMED_Selection_02.pivot_table(values = "pat_unproj", 
                                                        index = pack_id, columns = Label, 
                                                        aggfunc = sum, fill_value = 0)
    # COMED_Selection_03_columns = COMED_Selection_03.columns.tolist()
    COMED_Selection_03.columns = ["LB_" + x for x in COMED_Selection_03.columns.tolist()]
    COMED_Selection_tot2 = COMED_Selection_tot.pivot_table(values = "pat_unproj",
                                                           columns = Label, aggfunc = sum,
                                                           fill_value = 0)
    COMED_Selection_tot2.columns = ["tot_" + x for x in COMED_Selection_tot2.columns.tolist()]

    COMED_Selection_03["Total_market"] = COMED_Selection_03.sum(axis=1)

    q = "select * from COMED_Selection_tot2, COMED_Selection_03"
    COMED_Selection_04 = pysqldf(q)
    
    COMED_Selection_4model = discrimination_input(COMED_Selection_tot = COMED_Selection_tot,
                                                  COMED_Selection_04 = COMED_Selection_04,
                                                  label=Label)

    COMED_Selection_4model.sort_values(by = "ratio1", ascending = False, inplace = True)

    q = """
        select distinct """ + pack_id + """, line
        from knmp_univ1
        where """ + pack_id + """ in (select distinct """ + pack_id + """ from COMED_Selection_4model)
        """
    selected_line4model = pysqldf(q)
    print("MACRO: Selecting_Comedications ending...")
    return selected_line4model



def discrimination_input(COMED_Selection_tot,COMED_Selection_04,label,
                         minperc_indication=0.05, minperc_ratio1=0.01):
    print("MACRO: discrimination_input starting...")
    
    COMED_Selection_tot.drop_duplicates(subset = [label], inplace=True)
    globals()[label + "_list"] = COMED_Selection_tot.sort_values(by = label)
    globals()[label + "_list2"] = globals()[label + "_list"].reset_index(drop=True).reset_index()
    
    COMED_Selection_05 = COMED_Selection_04.fillna(0)
    
    filter_col = [col for col in COMED_Selection_04 if col.startswith('tot_')]
    COMED_Selection_05["perc_tot"] = COMED_Selection_05["Total_market"]/(COMED_Selection_05[filter_col].sum(axis=1))
    
    for i in range(COMED_Selection_tot[label].nunique()):
        COMED_Selection_05["perc_" + COMED_Selection_tot[label][i]] = COMED_Selection_05["LB_" + COMED_Selection_tot[label][i]]/COMED_Selection_05["tot_" + COMED_Selection_tot[label][i]]
        COMED_Selection_05["dif_perc_" + COMED_Selection_tot[label][i]] = COMED_Selection_05["perc_" + COMED_Selection_tot[label][i]]-COMED_Selection_05["perc_tot"]
        COMED_Selection_05["sq_" + COMED_Selection_tot[label][i]] = COMED_Selection_05["dif_perc_" + COMED_Selection_tot[label][i]]**2
        COMED_Selection_05["min_perc_" + COMED_Selection_tot[label][i]] =  (COMED_Selection_05["perc_" + COMED_Selection_tot[label][i]] > minperc_indication)
    
    COMED_Selection_4model = COMED_Selection_05.copy()
    
    filter_col = [col for col in COMED_Selection_4model if col.startswith('sq')]
    COMED_Selection_4model["ratio1"] = np.sqrt(COMED_Selection_4model[filter_col].sum(axis=1))
    COMED_Selection_4model["ratio2"] = COMED_Selection_4model["ratio1"]*COMED_Selection_4model["Total_market"]
    
    filter_col = [col for col in COMED_Selection_4model if col.startswith('dif_perc')]
    COMED_Selection_4model["ratio3"] = COMED_Selection_4model[filter_col].max(axis=1) - COMED_Selection_4model[filter_col].min(axis=1)
    
    COMED_Selection_4model = COMED_Selection_4model[COMED_Selection_4model["ratio1"]>minperc_ratio1]
    filter_col = [col for col in COMED_Selection_4model if col.startswith('min_perc_')]
    COMED_Selection_4model = COMED_Selection_4model[COMED_Selection_4model[filter_col].sum(axis=1) > 0]
    
    print("MACRO: discrimination_input ending...")
    return COMED_Selection_4model
    
def create_data4model_data2pred(features=None, focus_indication=None, model_nr=1, export="Y", pydir=None):
    focus_indication = focus_indication[["pat","indication"]]
    focus_indication.sort_values(by="pat", inplace=True)
    focus_indication.drop_duplicates(subset="pat", inplace=True)
    indication_features = pd.merge(left=features, right=focus_indication, how="left", on = "pat")
    data4modeling = indication_features[indication_features["indication"].notnull()]
    data2pred= indication_features[indication_features["indication"].isnull()]
    print("data4modeling shape: {}\ndata2pred shape: {}".format(data4modeling.shape, data2pred.shape))
    if export == "Y":
        print("exporting data4modeling/data2pred for model "+ str(model_nr))
        data4modeling.to_csv(pydir + "/data4modeling_m" + str(model_nr) + ".csv")
        data2pred.to_csv(pydir + "/data2pred" + str(model_nr) + ".csv")
    return data4modeling, data2pred
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    