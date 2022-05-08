from django.shortcuts import render
from sklearn.model_selection import train_test_split
import pandas as p
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier 
import joblib
from sklearn.model_selection import train_test_split

#importation dhouha
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create your views here.

travel=0
holidays=0
contact=0
know_well=0
hometime=0
strangers=0
inner_world=0
inadequate=0
sptime=0
sorry_end=0
ignore_difference=0
begin_correct=0
stress=0
anxious=0
likes=0
currentstress=0
goodtoleavehome=0
idntwhatsgoon=0
incompetence=0
fav_food=0
harmony=0
love=0
dreams=0
happy=0
trust=0
hope_wishes=0
freomvalue=0
commongoal=0
roles=0
harmony=0
entertain=0
hisitant_ind=0
friends_social=0
sudden_disscussion=0
care_sick=0
peoplegoal=0
negative_personality=0
notcalm=0
agro_argue=0
silent_for_calm=0
silence_instead_argue=0
offensive_expression=0
hate_subject=0
insult=0
humilate=0
calm_break=0
silence_fear=0
arguethenleave=0
accusations=0
imright=0
silence_fear_anger=0
mariage=0
guilty=0
wrong=0
hisitant_ind=0
always_never=0

age = 0
genre = 0
s1 = 0
s2 = 0
s4 = 0
s5 = 0
s6 = 0
s7 = 0
s8 = 0
s9 = 0
s10 = 0
s11 = 0
s12 = 0
s13 = 0
s14 = 0
s15 = 0
s16 = 0
s17 = 0
s18 = 0
s19 = 0
s20 = 0
s21 = 0
burnRes = 0


def index(request):

    context = {'a': 'hello'}
    return render(request, 'results.html', context)


def stress(request):
    context = {'a': 'hello'}
    if request.method == 'POST':
        return render(request, 'index.html', context)

    return render(request, 'formulaireStress.html', context)


def burnOut(request):
    context = {'a': 'hello'}
    
    if request.method == 'POST':
        
        return render(request, 'index.html', context)
    
    return render(request, 'formulaireBurnOut.html', context)


def burn1(request):
    global age
    global genre
    global s1
    global s2
    global s4
    global s5
    global s6
    context = {'a': 'hello'}
    if request.method == 'POST':
        age = int(request.POST['age'])
        genre = int(request.POST['genre'])
        s1 = int(request.POST['s1'])
        s2 = int(request.POST['s2'])
        s4 = int(request.POST['s4'])
        s5 = int(request.POST['s5'])
        s6 = int(request.POST['s6'])
        print("************************************************",
              age, type(s1), s2, s4, s5, s6)
        context = {'a': 'hello'}
        return render(request, 'formBurnout2.html', context)
    return render(request, 'formulaireBurnOut.html', context)


def burn2(request):
    global s7
    global s8
    global s9
    global s10
    global s11
    global s12
    global s13
    global s14
    context = {'a': 'hello'}
    if request.method == 'POST':
        print(age)
        s7 = int(request.POST['s7'])
        s8 = int(request.POST['s8'])
        s9 = int(request.POST['s9'])
        s10 = int(request.POST['s10'])
        s11 = int(request.POST['s11'])
        s12 = int(request.POST['s12'])
        s13 = int(request.POST['s13'])
        s14 = int(request.POST['s14'])
        print("************************************************",
              age, type(s7), s8, s9, s10, s11, s12, s13, s14)

        return render(request, 'formBurnout3.html', context)
    return render(request, 'formBurnout2.html', context)


def burn3(request):
    global s15
    global s16
    global s17
    global s18
    global s19
    global s20
    global s21
    global burnRes
    context = {'a': 'hello'}
    if request.method == 'POST':
        print(age)
        s15 = int(request.POST['s15'])
        s16 = int(request.POST['s16'])
        s17 = int(request.POST['s17'])
        s18 = int(request.POST['s18'])
        s19 = int(request.POST['s19'])
        s20 = int(request.POST['s20'])
        s21 = int(request.POST['s21'])
        print("************************************************",
              age, type(s15), s16, s17, s18, s19, s20, s21)
        burnRes = burnData()
        if(burnRes == 1):
            return render(request, 'resBurnoutPos.html', context)
        return render(request, 'resBurnoutNeg.html', context)
    return render(request, 'formBurnout3.html', context)


def burnData():
    df = p.read_csv("D:/ETUDE/FIA2/sem2/tp/Base avancé/survey.csv")
    # print(df['Country'].value_counts())
    # print("\n \n")
    # print(df['state'].unique())(data.info())

    # éliminer s colonnes et analyser la colonne age
    df.drop(columns=['Timestamp', 'Country',
            'state', 'comments'], inplace=True)
    df.drop(df[df['Age'] < 0].index, inplace=True)
    df.drop(df[df['Age'] > 100].index, inplace=True)
    print(df['Age'].unique())

    # unifier les valeurs
    df['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                          'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                          'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make', ], 'Male', inplace=True)

    df['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                          'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                          'woman', ], 'Female', inplace=True)

    df["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                          'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                          'Agender', 'A little about you', 'Nah', 'All',
                          'ostensibly male, unsure what that really means',
                          'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                          'Guy (-ish) ^_^', 'Trans woman', ], 'Other', inplace=True)

    print(df['Gender'].value_counts())

    print(df.isna().sum())

    # remplacer les valeurs nulle
    df['work_interfere'] = df['work_interfere'].fillna('Don\'t know')
    print(df['work_interfere'].unique())

    df['self_employed'] = df['self_employed'].fillna('No')
    print(df['self_employed'].unique())

    print(df.isna().sum())

    # afficher ts les valeurs pour cha colonnnes
    list_col = ['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
               'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                'benefits', 'care_options', 'wellness_program', 'seek_help',
                'anonymity', 'leave', 'mental_health_consequence',
                'phys_health_consequence', 'coworkers', 'supervisor',
                'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence']
    for col in list_col:
        print('{} :{} ' . format(col.upper(), df[col].unique()))

        # encoder les colonnes remplacer yes, no maybe par 0,1,2
        from sklearn.preprocessing import LabelEncoder
    object_cols = ['Gender', 'self_employed', 'family_history', 'treatment',
                   'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                   'benefits', 'care_options', 'wellness_program', 'seek_help',
                   'anonymity', 'leave', 'mental_health_consequence',
                   'phys_health_consequence', 'coworkers', 'supervisor',
                   'mental_health_interview', 'phys_health_interview',
                   'mental_vs_physical', 'obs_consequence']
    label_encoder = LabelEncoder()
    for col in object_cols:
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])

        # afficher les colonnes codés résultat
        list_col = ['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
                    'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                    'benefits', 'care_options', 'wellness_program', 'seek_help',
                    'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']
    for col in list_col:
        print('{} :{} ' . format(col.upper(), df[col].unique()))

        # Composer data ta en train et target
    X = df.drop('treatment', axis=1)
    y = df['treatment']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.3,
                                                        random_state=101)
    # analyser
    print("********************************************************************", age, genre, s1,
          s2, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21)
    print(df.info())
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    print("*******************************************************", X.info())
    preds = gb.predict([[age, genre, s1, s2, s4, s5, s6, s7, s8,
                         s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21]])
    return(preds[0])

    #print("Accuracy score : ", metrics.accuracy_score(y_test, preds))




def divtest1(request):
    global travel
    global holidays
    global contact
    global know_well
    global hometime
    global strangers
    global inner_world
    global inadequate
    global sptime
    context={}
    if request.method=="POST":
        travel=request.POST['travel']
        holidays=request.POST['holidays']
        contact=request.POST['contact']
        know_well=request.POST['know_well']
        hometime=request.POST['hometime']
        strangers=request.POST['strangers']
        inner_world=request.POST['inner_world']
        inadequate=request.POST['inadequate']
        sptime=request.POST['sptime']
        return render(request,'divorccetest2.html',context)
    return render(request,'formulaireDivorcetest1.html',context)
def divtest2(request):
    global sorry_end
    global ignore_difference
    global begin_correct
    global stress
    global anxious
    global currentstress
    global goodtoleavehome
    global idntwhatsgoon
    global incompetence
    context={}
    if request.method=="POST":
        sorry_end=request.POST['sorry_end']
        ignore_difference=request.POST['ignore_difference']
        begin_correct=request.POST['begin_correct']
        stress=request.POST['stress']
        anxious=request.POST['anxious']
        currentstress=request.POST['currentstress']
        goodtoleavehome=request.POST['goodtoleavehome']
        idntwhatsgoon=request.POST['idntwhatsgoon']
        incompetence=request.POST['incompetence']
        return render(request,'divorcetest3.html',context)
    return render(request,'divorcetest2.html',context)
def divtest3(request):
    global fav_food
    global harmony
    global love
    global dreams
    global happy
    global trust
    global hope_wishes
    global freomvalue
    global commongoal
    context={}
    print('*******************************************',travel)
    if request.method=="POST":
        print('*******************************************',travel)
        fav_food=request.POST['fav_food']
        harmony=request.POST['harmony']
        love=request.POST['love']
        dreams=request.POST['dreams']
        happy=request.POST['happy']
        trust=request.POST['trust']
        hope_wishes=request.POST['hope_wishes']
        freomvalue=request.POST['freomvalue']
        commongoal=request.POST['commongoal']
        return render(request,'divorcetest4.html',context)
    return render(request,'divorcetest3.html',context)
def divtest4(request):
    global roles
    global entertain
    global friends_social
    global sudden_disscussion
    global care_sick
    global peoplegoal
    global negative_personality
    global likes
    global harmony
    global hisitant_ind
    context={}
    if request.method=="POST":
        roles=request.POST['roles']
        entertain=request.POST['entertain']
        friends_social=request.POST['friends_social']
        sudden_disscussion=request.POST['sudden_disscussion']
        care_sick=request.POST['care_sick']
        peoplegoal=request.POST['peoplegoal']
        negative_personality=request.POST['negative_personality']
        likes=request.POST['likes']
        harmony=request.POST['harmony']
        hisitant_ind=request.POST['hisitant_ind']
        return render(request,'divorcetest5.html',context)
    return render(request,'divorcetest4.html',context)
def divtest5(request):
    global notcalm
    global agro_argue
    global silent_for_calm
    global silence_instead_argue
    global offensive_expression
    global hate_subject
    global insult
    global humilate
    global calm_break
    global silence_fear
    global arguethenleave
    context={}
    if request.method=="POST":
        notcalm=request.POST['notcalm']
        agro_argue=request.POST['agro_argue']
        silent_for_calm=request.POST['silent_for_calm']
        silence_instead_argue=request.POST['silence_instead_argue']
        offensive_expression=request.POST['offensive_expression']
        hate_subject=request.POST['hate_subject']
        insult=request.POST['insult']
        humilate=request.POST['humilate']
        calm_break=request.POST['calm_break']
        silence_fear=request.POST['silence_fear']
        arguethenleave=request.POST['arguethenleave']
        return render(request,'divorcetest6.html',context)
    return render(request,'divorcetest5.html',context)
def divtest6(request):
    global accusations
    global imright
    global silence_fear_anger
    global mariage
    global guilty
    global wrong
    global always_never
    print (travel)
    context={}
    if request.method=="POST":
        accusations=request.POST['accusations']
        imright=request.POST['imright']
        silence_fear_anger=request.POST['silence_fear_anger']
        mariage=request.POST['mariage']
        guilty=request.POST['guilty']
        wrong=request.POST['wrong']
        always_never=request.POST['always_never']
        model=joblib.load('divorce_model.joblib')
        data_test=[sorry_end,ignore_difference,begin_correct,contact,sptime,hometime,strangers,holidays,travel,commongoal,harmony,freomvalue,entertain,peoplegoal,dreams,love,happy,mariage,roles,trust,likes,care_sick,fav_food,stress,inner_world,anxious,currentstress,hope_wishes,know_well,friends_social,agro_argue,always_never,negative_personality,offensive_expression,insult,humilate,notcalm,hate_subject,sudden_disscussion,idntwhatsgoon,calm_break,arguethenleave,silent_for_calm,goodtoleavehome,silence_instead_argue,silence_fear,silence_fear_anger,imright,accusations,guilty,wrong,hisitant_ind,inadequate,incompetence]
        print(data_test)
        preds=model.predict([data_test])
        print(preds[0])
        if preds[0]== 1:
           return render(request,'results.html',context)
        else:
           return render(request,'conseil.html',context)
    return render(request,'divorcetest6.html',context)






def stress_form1(request):
    global age2
    global genre2
    global married,enfants,durable,gained,membres_familles,niveau,saveasset,living_expenses,durable
    context = {'a': 'hello'}
    if request.method == 'POST':
        age2 = int(request.POST['age'])
        genre2 = int(request.POST['genre'])
        married = int(request.POST['married'])
        enfants = int(request.POST['enfants'])
        niveau = int(request.POST['niveau'])
        membres_familles = int(request.POST['membres_familles'])
        gained = int(request.POST['gained'])
        durable = int(request.POST['durable'])
        saveasset = int(request.POST['saveasset'])
        living_expenses = int(request.POST['living_expenses'])

        print("************************************************",
              age,genre , married , enfants,niveau,membres_familles,gained,durable,saveasset,living_expenses)
        context = {'a': 'hello'}
        return render(request, 'formulaireStress2.html', context)
    return render(request, 'formulaireStress.html', context)

    
def stress_form2(request):
    global others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural,farm_expenses,labor_primary,lasting_investment,no_lasting_investment
    context = {'a': 'hello'}
    if request.method == 'POST':
        others_expenses = int(request.POST['others_expenses'])
        incoming_salary = int(request.POST['incoming_salary'])
        incoming_own_farm = int(request.POST['incoming_own_farm'])
        incoming_business = int(request.POST['incoming_business'])
        incoming_no_business = int(request.POST['incoming_no_business'])
        incoming_agricultural = int(request.POST['incoming_agricultural'])
        farm_expenses = int(request.POST['farm_expenses'])
        labor_primary = int(request.POST['labor_primary'])
        lasting_investment = int(request.POST['lasting_investment'])
        no_lasting_investment = int(request.POST['no_lasting_investment'])

        print("***** Données du test stress NIVEAU 2 *****",others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural
        ,farm_expenses,labor_primary,lasting_investment,no_lasting_investment)
        res_stress = stressData()
        if(res_stress == 1):
            return render(request, 'res_test_stress_neg.html', context)
        return render(request, 'res_test_stress_neg.html', context)
    return render(request, 'formulaireStress2.html', context)

def stressData():
    Depression=pd.read_csv("C:/Users/attia/Downloads/archive/b_depressed.csv")
    #Depression
    Depression.head()
    Depression.no_lasting_investmen.value_counts()
    Depression.describe()
    Depression = Depression[~(Depression.no_lasting_investmen.isnull())]
    Depression = Depression.drop(['Survey_id','Ville_id'],axis=1)
    Depression.info()
    Depression.depressed.value_counts()
    """
    # Pour régler la taille de l'image
    plt.figure(figsize=(13,7))
    # bsh torsem l matrice lkbira , fiha les colonnes lkol maa les valeurs , cmap hua lcouleur li khtartou
    #wel annot=true bch yhotlk les valeurs
    sns.heatmap(Depression.corr(),annot=True,cmap="YlGnBu")
    plt.title('Heatmap of Variable Correlations',fontsize=25)
    plt.show()
    
    #bsh n9arnou bin lm3arsinn w ldepression
    plt.figure(figsize=(15,5))
    sns.barplot(x='Married',y='depressed',data=Depression, palette = "Blues")
    plt.title('Married vs Depp')
    plt.xlabel('Married')
    plt.ylabel('Depressed')
    plt.show()
    # => Elly mahomsh m3arsin dépressé akthr ml m3arsiin

    #Gender : 0 mra w 1 : rajel
    #bsh n9arnou depression bin lwomam o l man w l m3arsin wlaa
    plt.figure(figsize=(15,5))
    sns.barplot(x='sex',y='depressed',hue='Married',data=Depression,palette = "Blues")
    plt.title('Depressed vs Marital Status vs Gender')
    plt.xlabel('Gender')
    plt.ylabel('Depressed')
    plt.show()
    #Resultat :Unmarried men are more depressed than married men. 
    # Married women are more depressed than unmarried women. 
    
    #bsh n9arnou bin number te3 children w depressed
    plt.figure(figsize=(15,5))
    sns.barplot(x='depressed',y='Number_children',data=Depression, palette = "Blues")
    plt.title('Depressed vs Number of Children')
    plt.xlabel('Depressed')
    plt.ylabel('Number of Children')
    plt.show()
    #famesh yecer far9 car nes lkol andha nbr te3 children entre 1 et 3 , maaaax 4

    #n9arnou bin education level o depression
    plt.figure(figsize=(15,5))
    sns.barplot(x='depressed',y='education_level',data=Depression, palette = "Blues")
    plt.title('Depressed vs Education level')
    plt.xlabel('Depressed')
    plt.ylabel('Education level')
    plt.show()
    #Result : educated individuals are less deprresed.

    #n9arnou bin incoming salary o depression
    plt.figure(figsize=(15,5))
    sns.barplot(x='depressed',y='incoming_salary',data=Depression, palette = "Blues")
    plt.title('Depressed vs Incoming Salary')
    plt.xlabel('Depressed')
    plt.ylabel('Incoming Salary')
    plt.show()
    """
    #=>famesh far9 kbiiiir

         # Composer data ta en train et target
    X = Depression.drop('depressed', axis=1)
    y = Depression['depressed']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.3,
                                                        random_state=101)
    X_train.head()  
    y_train.head()
    Model = RandomForestClassifier(random_state=100,n_jobs=-1)

    params = {'max_depth':[3,5,10,12,15],
            'min_samples_leaf':[60,70,80,90,100,120,130],
            'n_estimators':[200], 
            'max_features': [0.05,0.1,0.15,0.2],
            'criterion': ["gini"]}

    grid_search = GridSearchCV(param_grid=params,estimator=Model,n_jobs=-1,verbose=1,scoring='recall')
    grid_search.fit(X_train,y_train)
    from sklearn.metrics import accuracy_score, recall_score, plot_roc_curve
    
    Model.fit(X_train,y_train)

    print(Depression.info())
    Model = RandomForestClassifier()
    Model.fit(X_train, y_train)
    print("*******************************************************", X.info())
    preds = Model.predict([[age2,genre2,married,enfants,niveau,membres_familles,
    gained,durable,saveasset,living_expenses,others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural,farm_expenses,
    labor_primary,lasting_investment,no_lasting_investment]])

    return(preds[0])
    #accuracy 96
    plot_roc_curve(Model,X_train,y_train)
    plt.show()
    y_train_pred = Model.predict([[age2,genre2,married,enfants,niveau,membres_familles,
    gained,durable,saveasset,living_expenses,others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural,farm_expenses,
    labor_primary,lasting_investment,no_lasting_investment,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    print("Accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Recall: ", recall_score(y_train, y_train_pred))

    #accuracy 77
    plot_roc_curve(Model,X_test,y_test)
    plt.show()
    y_test_pred = Model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Recall: ", recall_score(y_test, y_test_pred))

    """
    preds = Model.predict([[age2,genre2,married,enfants,niveau,membres_familles,
    gained,durable,saveasset,living_expenses,others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural,farm_expenses,
    labor_primary,lasting_investment,no_lasting_investment]])
    print(preds[0])
    return(preds[0])
    

    print("*******************************************************", X.info())
    preds = Model.predict([[age2,genre2,married,enfants,niveau,membres_familles,
    gained,durable,saveasset,living_expenses,others_expenses,incoming_salary,incoming_own_farm,incoming_business,incoming_no_business,incoming_agricultural,farm_expenses,
    labor_primary,lasting_investment,no_lasting_investment]])
    print(preds[0])
    return(preds[0])

    """
