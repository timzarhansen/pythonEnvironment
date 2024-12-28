import pandas as pd
import numpy as np
def read_csv(file_path):
    data =  pd.read_csv(file_path, sep=';', header=0)
    return data

# Bremen 19
file_path = 'data/gesamtergebnis_19_Bremen.csv'
df = read_csv(file_path)

dataBremen19 = df.loc[:,["gebiet-nr","gebiet-name","B","D3_LISTE","D3_SUMME_KANDIDATEN","D2","C","D1"]]
# df.loc[["gebiet-nr","gebiet-name","D3_LISTE","B"],:]
pd.set_option('display.max_columns', None)
# print(dataBremen19.head())

# Bremen 23
file_path = 'data/gesamtergebnis_23_Bremen.csv'
df = read_csv(file_path)
print(df.head())
dataBremen23 = df.loc[:,["Bezirksnummer","Stimmen gueltige (D2)","D03 Personenstimmen","D03 Listenstimmen"]]

print(dataBremen23.head())
print(dataBremen23.loc[4:6,:])


df = pd.read_excel('data/Strassenverzeichnis_Bremen.xlsx',sheet_name="Straßenabschnitt")
# print(df.head())
strassenverzeichnis = df.loc[:,["Straßen, Wege, Plätze\n(amtliche Bezeichnung)","ungerade Hnr.","Unnamed: 4","gerade Hnr.","Unnamed: 8","Wahlbezirk"]]
strassenverzeichnis = strassenverzeichnis.rename(columns={"Straßen, Wege, Plätze\n(amtliche Bezeichnung)":"Straße"})
strassenverzeichnis = strassenverzeichnis.rename(columns={"ungerade Hnr.":"ungerade Hnr. von"})
strassenverzeichnis = strassenverzeichnis.rename(columns={"Unnamed: 4":"ungerade Hnr. bis"})
strassenverzeichnis = strassenverzeichnis.rename(columns={"gerade Hnr.":"gerade Hnr. von"})
strassenverzeichnis = strassenverzeichnis.rename(columns={"Unnamed: 8":"gerade Hnr. bis"})
strassenverzeichnis = strassenverzeichnis.rename(columns={"Wahlbezirk":"Wahlbezirk"})

print("strassenverzeichnis: ")
print(strassenverzeichnis.head())
columns = ['Straße', 'ungerade Hnr. von', 'ungerade Hnr. bis', 'gerade Hnr. von', 'gerade Hnr. bis', '19', '23']
# resultingListOfOutput = pd.DataFrame(columns=['Straße', 'ungerade Hnr. von', 'ungerade Hnr. bis', 'gerade Hnr. von', 'gerade Hnr. bis', '19', '23'])
data = {col: [None] * strassenverzeichnis.shape[0] for col in columns}
resultingListOfOutput = pd.DataFrame(data)
streetIndex = 1
while streetIndex<strassenverzeichnis.shape[0]:
    streetName = strassenverzeichnis.loc[streetIndex,"Straße"]
    wahlBezirk = strassenverzeichnis.loc[streetIndex, "Wahlbezirk"]
    try:
        #23
        currentDataBremen23 = dataBremen23.loc[dataBremen23['Bezirksnummer'] == wahlBezirk]
        stimmenGesammt23 = currentDataBremen23["D03 Listenstimmen"]+currentDataBremen23["D03 Personenstimmen"]
        percentageGreenDirect23 = stimmenGesammt23.astype(float)/(currentDataBremen23["Stimmen gueltige (D2)"].astype(float))
        percentageGreenDirect23 = float(percentageGreenDirect23.to_numpy())
        # 19
        currentDataBremen19 = dataBremen19.loc[dataBremen19['gebiet-nr'] == wahlBezirk]
        stimmenGesammt19 = currentDataBremen19["D3_LISTE"].astype(float) + currentDataBremen19["D3_SUMME_KANDIDATEN"].astype(float)
        percentageGreenDirect19 = (stimmenGesammt19) / (currentDataBremen19["D2"].astype(float))
        percentageGreenDirect19 = float(percentageGreenDirect19.to_numpy())

        # New row data
        new_row = {'Straße': streetName, 'ungerade Hnr. von': strassenverzeichnis.loc[streetIndex, "ungerade Hnr. von"], 'ungerade Hnr. bis': strassenverzeichnis.loc[streetIndex, "ungerade Hnr. bis"], 'gerade Hnr. von': strassenverzeichnis.loc[streetIndex, "gerade Hnr. von"],
                   'gerade Hnr. bis': strassenverzeichnis.loc[streetIndex, "gerade Hnr. bis"], '19': percentageGreenDirect19, '23': percentageGreenDirect23}


        resultingListOfOutput.loc[streetIndex] = new_row


        greenColor= percentageGreenDirect19*255*2
        occupancy = 0.7
        redColor = 0
        blueColor = 0

        # resultingListOfOutput.loc[]
        # print("streetName: ",streetName)
        # print("wahlBezirk: ",wahlBezirk)
        # print("percentageGreenDirect19: ", percentageGreenDirect19)
        # print("percentageGreenDirect23: ", percentageGreenDirect23)
    except Exception as error:
        print("An exception occurred:", error)

    streetIndex = streetIndex+1

print("test")

resultingListOfOutput.to_csv("tmp/resultsElectionStreet.csv", sep=';', encoding='utf-8')











