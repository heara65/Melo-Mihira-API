# import json
# import pickle
# import pandas as pd
# import sklearn

# with open("Kmaensmodel1.pkl", "rb") as f:
#     model = pickle.load(f)

# PredictedDataset = pd.read_csv('FinalDataset.csv',
#                                encoding='unicode_escape')

# ActivityDataset = pd.read_csv('ActivityRecommendation_v3.csv',
#                               encoding='unicode_escape')
# # Encoding data

# genderdict = {'m': 0, 'f': 1}

# countrylist = ['United States', 'Germany', 'United Kingdom', 'Poland', 'Russian Federation', 'Brazil', 'Sweden',
#                'Spain', 'Finland', 'Netherlands', 'Italy', 'France', 'Canada', 'Australia', 'Turkey', 'Norway',
#                'Czech Republic', 'Ukraine', 'Japan', 'Belgium', 'Mexico', 'Argentina', 'Switzerland', 'Austria',
#                'Romania', 'Portugal', 'Bulgaria', 'Chile', 'Denmark', 'Colombia', 'Greece', 'Hungary', 'Latvia',
#                'Slovakia', 'Croatia', 'Serbia', 'Lithuania', 'Estonia', 'Ireland', 'New Zealand', 'Belarus', 'Israel',
#                'India', 'Venezuela', 'Indonesia', 'Singapore', 'Slovenia', 'Korea, Republic of', 'China',
#                'South Africa', 'Malaysia', 'Philippines', 'Peru', 'Thailand', 'Moldova', 'Costa Rica', 'Iceland',
#                'Taiwan', 'Paraguay', 'Bosnia and Herzegovina', 'Antarctica', 'Puerto Rico', 'Georgia', 'Macedonia',
#                'Uruguay', 'Honduras', 'Barbados', 'Kazakhstan', 'Andorra', 'Saudi Arabia',
#                'United States Minor Outlying Islands', 'Djibouti', 'Cocos (Keeling) Islands', 'Tunisia', 'Egypt',
#                'Bolivia', 'Panama', 'Brunei Darussalam', 'Iran, Islamic Republic of', 'Dominican Republic',
#                'El Salvador', 'Haiti', 'Ecuador', 'Guatemala', 'Morocco', 'Pakistan', 'Burkina Faso', 'Azerbaijan',
#                'Cambodia', 'Hong Kong', 'Viet Nam', 'United Arab Emirates', 'Jamaica', 'Faroe Islands', 'Somalia',
#                'Guinea-Bissau', 'Micronesia, Federated States of', 'Tuvalu', "Cote D'Ivoire", 'Libyan Arab Jamahiriya',
#                'Nicaragua', 'Kyrgyzstan', 'Malta', 'Bermuda', 'Luxembourg', 'Kuwait', 'Cyprus',
#                'Heard Island and Mcdonald Islands', 'Christmas Island', 'Cuba', 'Niue', 'Aruba', 'Vanuatu', 'Dominica',
#                'Holy See (Vatican City State)', 'Uzbekistan', 'Bhutan', 'Montenegro', 'Reunion', 'Fiji',
#                'Netherlands Antilles', 'Lebanon', 'Liechtenstein']
# countrydict = dict(zip(countrylist, [i for i in range(len(countrylist))]))

# ethnicityList = ['white', 'black', 'asian', 'indian', 'others']
# ethnicitydict = dict(zip(ethnicityList, [i for i in range(len(ethnicityList))]))


# # print(countrydict)

# # Define a mapper functions
# def mapr1(key):
#     """ Maps numbers to categories (gender)"""
#     return genderdict[key]


# def mapr2(key):
#     """ Maps numbers to categories (country)"""
#     return countrydict[key]


# def mapr3(key):
#     """ Maps numbers to categories (country)"""
#     return ethnicitydict[key]


# def recommendDemographics(age, gender, Ethnicity):
#     GenderEncoded = mapr1(gender)
#     EthnicityEncoded = mapr3(Ethnicity)
#     S = [[age, EthnicityEncoded, GenderEncoded]]
#     dfTest = pd.DataFrame(S, columns=['age', 'Ethnicity_Encoded', 'Gender_Encoded'])
#     clustering = model.predict(dfTest)
#     Grouped = PredictedDataset.loc[PredictedDataset['Cluster'] == clustering[0]]
#     sorted = Grouped.sort_values(by=['plays'], ascending=False)
#     topK = sorted.head(10)
#     topkArtists = topK['artist'].tolist()
#     # jsonStr = json.dumps(topkArtists)
#     # return jsonStr
#     return topkArtists


# def recommendEmotion(emotion):
#     Grouped = PredictedDataset
#     EmotionList = ['Happy', 'Neutral', 'Angry', 'Sad']
#     if emotion == EmotionList[0]:
#         EmotionGrouped = Grouped.loc[PredictedDataset['SongEmotionScore.positive_score'] != 0]
#     elif emotion == EmotionList[1]:
#         EmotionGrouped = Grouped.loc[PredictedDataset['SongEmotionScore.neutral_score'] != 0]
#     elif emotion == EmotionList[2]:
#         EmotionGrouped = Grouped.loc[PredictedDataset['SongEmotionScore.negative_score'] != 0]
#     elif emotion == EmotionList[3]:
#         EmotionGrouped1 = Grouped.loc[PredictedDataset['SongEmotionScore.negative_score'] != 0]
#         EmotionGrouped = EmotionGrouped1.loc[PredictedDataset['SongEmotionScore.neutral_score'] != 0]
#     EmotionGroupedBySumArtist = EmotionGrouped.groupby(['artist'])['plays'].agg('sum')
#     ToppKArtists = EmotionGroupedBySumArtist.nlargest(15)
#     ArtistList = ToppKArtists.index.values.tolist()
#     # jsonStr = json.dumps(ArtistList)
#     return ArtistList


# def recommendActivity(activity):
#     # Grouped = ActivityDataset
#     Grouped = ActivityDataset.loc[ActivityDataset['activity'] == activity]
#     sorted = Grouped.sort_values(by=['plays'], ascending=False)
#     topK = sorted.head(10)
#     topkArtists = topK['artist'].tolist()
#     res2 = []
#     [res2.append(x) for x in topkArtists if x not in res2]
#     return res2
