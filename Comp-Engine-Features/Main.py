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
print("*** Done ***")



# Reading Required data eg.Time series and hctsa data information

warnings.filterwarnings("ignore",category=RuntimeWarning)
print("Reading Time series data")

Alltimeseries=[]
with open('hctsa_timeseries-data.csv','r') as read_obj:
    csv_reader=reader(read_obj)
    li=list(csv_reader)
    for i in tqdm(li):
        Alltimeseries.append(list(map(float,i)))
print("\n")


print("Reading Hctsa data matrix ")
hctsa=pd.read_csv('hctsa_datamatrix.csv')
keywords=pd.read_csv('hctsa_features.csv')


functionname = input("Enter Function name (same as defined in your .py file) -:  ")
to_zscore = input("Want timeseries to be z-scored before passing through your feature ? (y/n) : \n ").lower()
New_feature_vector=[]
Invalid_Ts=[]


if to_zscore == 'y':
    for i in range(len(Alltimeseries)):
        z_timeseries = pd.DataFrame(Alltimeseries[i]).apply(zscore)
        try:
            featurevalue=getattr(UserFunction,functionname)(z_timeseries.values)
            New_feature_vector.append(featurevalue)
        except AttributeError as A:
            print(A)
            sys.exit()
        except:
            Invalid_Ts.append(i)
    print("*** Feature Vector generated ***")
elif to_zscore == 'n':
    for i in range(len(Alltimeseries)):
        try:
            featurevalue = getattr(UserFunction, functionname)(Alltimeseries[i])
            New_feature_vector.append(featurevalue)
        except AttributeError as A:
            print(A)
            sys.exit()
        except:
            Invalid_Ts.append(i)
    print("*** Feature Vector generated ***")
else:
    print("Wrong Input")
    sys.exit()
New_feature_vector=pd.DataFrame(New_feature_vector)



missing=[]
final=[]
def CheckMissing(featurematrix,threshold):
    for i in range(len(featurematrix.columns)):
        if (featurematrix.iloc[:,i].isna().sum()*100/featurematrix.iloc[:,i].shape[0])>threshold:
            missing.append(featurematrix.columns[i])
        else:
            final.append(i)
    print(f"Columns having more then {threshold} Missing values are : {len(missing)}")
    print(missing)
    return final

print("\n\n")
print("Removing columns Having more than 70% missing values.....\n")
finalfeatures=CheckMissing(hctsa,70)
print("done")


print("Finding correlation ....")

alpha=0.05
nan_fvector=int(New_feature_vector.isna().sum())
correlatedfeatures=[]
for i in finalfeatures:
    eachfeature=[]
    if (hctsa.iloc[:,i].isna().sum()+nan_fvector)<=50:
        coef, p = spearmanr(hctsa.iloc[:,i],New_feature_vector.values,nan_policy="omit")
        if p < alpha:
            eachfeature=[hctsa.columns[i],p,abs(coef),i,keywords.iloc[i:i+1,2].values,coef]
            correlatedfeatures.append(eachfeature)
print("\n")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


BestMatches=sorted(correlatedfeatures,key=itemgetter(2))[::-1]
DATAFRAME=pd.DataFrame(BestMatches)
DATAFRAME.columns=['Name','p-value','Corr','Column-Id','Keywords','Signed corr value']
DATAFRAME.to_csv('matching data.csv')
print(DATAFRAME[:10])


PairWise=pd.DataFrame()
PairWise[functionname]=New_feature_vector[0]
import matplotlib.pyplot as plt
print("\n")
PlotType=input(" Plot should be on 'Raw' values or 'Rank' based ?\n (rw/rn)").lower()
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
else:
    print("Invalid input")




pairwise_corr=PairWise.corr(method="spearman").abs()
g=sns.clustermap(pairwise_corr,method="complete",annot=True,linewidth=0.5)
from matplotlib.patches import Rectangle
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




