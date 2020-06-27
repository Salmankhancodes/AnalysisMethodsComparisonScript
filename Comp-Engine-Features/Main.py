print("Importing Required Modules....")
import pandas as pd
import warnings
import UserFunction
from csv import reader
from tqdm import tqdm
from scipy.stats import zscore
from scipy.stats import spearmanr
from operator import itemgetter
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)
print("*** Done ***")



Alltimeseries = []
missing = []
final = []

''' Reading all required files - timeseries data, hctsa datamatrix and keywords '''

print("Reading Time series data")
with open('hctsa_timeseries-data.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    li = list(csv_reader)
    for i in tqdm(li):
        Alltimeseries.append(list(map(float, i)))
print("\n")

print("Reading Hctsa data matrix ")
hctsa = pd.read_csv('hctsa_datamatrix.csv')
keywords = pd.read_csv('hctsa_features.csv')



def PassingTimeseriesToNewFeature(Alltimeseries):

   ###  Opt for zscoring befor timeseries ####

    to_zscore = input("Want timeseries to be z-scored before passing through your feature ? (y/n) : \n ").lower()


    New_feature_vector=[]    ###### to store value generated for each timeseries

    Invalid_Ts=[]           #####  to store index of time series that does not perform well


######   for processing z-scored timeseries ##########

    if to_zscore == 'y':
        for i in range(len(Alltimeseries)):
            z_timeseries = pd.DataFrame(Alltimeseries[i]).apply(zscore)
            try:
                featurevalue=getattr(UserFunction,functionname)(z_timeseries.values)
                New_feature_vector.append(featurevalue)
            except AttributeError as A:
                print(A)
                sys.exit()
            except TypeError as T:
                print(T)
                print("Please re-check your python code or try without z-score")
                sys.exit()
            except:
                Invalid_Ts.append(i)


#########   for processing raw timeseries  #################
    elif to_zscore == 'n':
        for i in range(len(Alltimeseries)):
            try:
                featurevalue = getattr(UserFunction, functionname)(Alltimeseries[i])
                New_feature_vector.append(featurevalue)
            except AttributeError as A:
                print(A)
                sys.exit()
            except TypeError as T:
                print(T)
                print("Please re-check your python code")
                sys.exit()
            except:
                Invalid_Ts.append(i)
    #############   for handling invalid input ###############3

    else:
        print("Wrong Input")
        sys.exit()


##############   for handling too many Nan values  ############


    if int(pd.DataFrame(New_feature_vector).isna().sum())>50:
        print("Too many Nan Values")
        sys.exit()



   #####   return feature vector and indexes of invalid time series   ###########
    return New_feature_vector,Invalid_Ts



#############  function for removing column having more missing values than threshold  ############


def CheckMissing(featurematrix,threshold):
    for i in range(len(featurematrix.columns)):
        if (featurematrix.iloc[:,i].isna().sum()*100/featurematrix.iloc[:,i].shape[0])>threshold:
            missing.append(featurematrix.columns[i])
        else:
            final.append(i)
    print(f"Columns having more then {threshold} Missing values are : {len(missing)}")
    print(missing)

    ###########   returns list indexes of appropriate columns  (few or no missing values) of hctsa       ############3
    return final



#####################   for comparing feature vector with each column of data matrix   ###############

def Comparison_With_Matrix(New_feature_vector,finalfeatures):
    alpha=0.05
    New_feature_vector=pd.DataFrame(New_feature_vector)

    ##############   calculating total Nan values in feature vector ##################
    nan_fvector=int(New_feature_vector.isna().sum())
    correlatedfeatures=[]
    for i in finalfeatures:
        eachfeature=[]

        ############  if sum of nan values with column and featurevector less than 50  #########

        if (hctsa.iloc[:,i].isna().sum()+nan_fvector)<50:
            ####  corr between columns and omitting bad pairs  #############
            coef, p = spearmanr(hctsa.iloc[:,i],New_feature_vector.values,nan_policy="omit")
            if p < alpha:

        ####  storing details of bestmatches in list ##########

                eachfeature=[hctsa.columns[i],p,abs(coef),i,keywords.iloc[i:i+1,2].values,coef]
                correlatedfeatures.append(eachfeature)
    BestMatches = sorted(correlatedfeatures, key=itemgetter(2))[::-1]
#######  return sorted bestmatches list of lists ###########
    return BestMatches


#####            Visualization          #####

def visualization(BestMatches, New_feature_vector):
    ######  creating dataframe of pairwise correlation  ####
    PairWise=pd.DataFrame()

    ######### appending user's feature vector to pairwise datframe ####
    PairWise[functionname]=New_feature_vector
    New_feature_vector = pd.DataFrame(New_feature_vector)

    print("\n")


    #####   opt for raw based scatter plot or rank based   #######


    PlotType=input(" Plot should be on 'Raw' values or 'Rank' based ?\n (rw/rn)").lower()


    ########   rank based scatter plot   ##########

    if PlotType=='rn':
        plt.figure(figsize=(40, 30))
        for i in range(len(BestMatches[:10])):
            plt.subplot(4, 4, i + 1)
            plt.subplots_adjust(left=0.07, bottom=0.00, right=0.94, top=0.92, wspace=0.35, hspace=0.68)
            plt.scatter(hctsa.iloc[:, BestMatches[i][3]].rank(), New_feature_vector.rank())
            PairWise[BestMatches[i][0]] = hctsa.iloc[:, BestMatches[i][3]]
            plt.title(f"Correlation = {BestMatches[i][2]}",fontsize=10)
            plt.xlabel(BestMatches[i][0])
            plt.ylabel(functionname)
        plt.show()


    ########   raw based scatter plot   ##########


    elif PlotType=='rw':
        plt.figure(figsize=(40, 30))
        plt.subplots_adjust(left=0.07,bottom=0.00,right=0.94,top=0.92,wspace=0.35,hspace=0.68)
        for i in range(len(BestMatches[:10])):
            plt.subplot(4, 4, i + 1)
            plt.scatter(hctsa.iloc[:,BestMatches[i][3]], New_feature_vector)
            PairWise[BestMatches[i][0]] = hctsa.iloc[:, BestMatches[i][3]]
            plt.title(f"Correlation = {BestMatches[i][2]}",fontsize=10)
            plt.xlabel(BestMatches[i][0])
            plt.ylabel(functionname)
        plt.show()


    #######3    Handling invalid option #######3
    else:
        print("Invalid input")

        sys.exit()
    ############    Reordering the resuult with Linkage clustering  ########


    pairwise_corr=PairWise.corr(method="spearman").abs()
    g=sns.clustermap(pairwise_corr,method="complete",annot=True,linewidth=0.5)


    #######      Highlighting user's row and column in pairwise correlation with patch    ########

    columns = list(PairWise.columns)

    N = len(columns)
    wanted_label = functionname
    wanted_row = g.dendrogram_row.reordered_ind.index(columns.index(wanted_label))
    wanted_col = g.dendrogram_col.reordered_ind.index(columns.index(wanted_label))

    xywh_row = (0, wanted_row, N, 1)
    xywh_col = (wanted_col, 0, 1, N)
    for x, y, w, h in (xywh_row, xywh_col):
        g.ax_heatmap.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='Blue', lw=4, clip_on=True))
    g.ax_heatmap.tick_params(length=0)
    plt.show()




#######   main function    ##########3

if __name__ == '__main__':


    #### taking user's function name as input  ########


    functionname = input("Enter Function name (same as defined in your .py file) -:  ")


    ###  calling function for generating feature vector ########

    New_feature_vector,Invalid_Ts=PassingTimeseriesToNewFeature(Alltimeseries)
    print(New_feature_vector)
    print(" *** Feature vector generated ***")


    print("Removing columns Having more than 70% missing values.....\n")

    ###  calling function for removing nan values from datamatrix    ########
    finalfeatures = CheckMissing(hctsa, 70)
    print("done")

    print("Finding correlation ....")

    ########   Removing rows from datamatrix which didn't perform well on user's function ###

    hctsa = hctsa.drop(hctsa.index[Invalid_Ts]).reset_index(drop=True)



    ###  calling function for finding correlation with  feature vector ########
    BestMatches=Comparison_With_Matrix(New_feature_vector,finalfeatures)


    #####  displaying all rows and columns  #####
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)


##########   Creating dataframe for All results and saving in CSV file   #######
    DATAFRAME = pd.DataFrame(BestMatches)
    DATAFRAME.columns = ['Name', 'p-value', 'Corr', 'Column-Id', 'Keywords', 'Signed corr value']
    DATAFRAME.to_csv('matching data.csv')


    #########   printing 10 best matches   ######
    print(DATAFRAME[:10])

############ visualization with scatter plot and Pairwise correlation  #####
    visualization(BestMatches, New_feature_vector)