#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Following packages are installed
# pip install folium
# pip install geopy
# pip install lxml
# pip install wget


# In[2]:


import lxml
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
from pandas.io.json import json_normalize
import folium
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
import requests
import wget


# # 1. Get Venue Data

# In[3]:


#import the data
url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
req=pd.read_html(url)
toronto=req[0]

df_toronto=toronto[toronto['Borough']!='Not assigned'].reset_index(drop=True)
df_toronto.head()


# In[4]:


geourl='http://cocl.us/Geospatial_data'
geodata=pd.read_csv(geourl)

df2=df_toronto.merge(geodata,on='Postal Code')
df2.head()


# In[5]:


city = 'Toronto'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(city)
lat = location.latitude
lon = location.longitude


map_toronto = folium.Map(location=[lat, lon], zoom_start=11)

# add markers to map
for lat, lng, label in zip(df2['Latitude'], df2['Longitude'], df2['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# # 2. Get Venue information from Foursquare

# In[10]:


CLIENT_ID = '2OMIVBGRDZ4GVNN4YNWCNZUZ5VJNHRYFMJOPWS5Y5E3D53QV' 
CLIENT_SECRET = 'V3E1N5UH5WVUM5DXEJPVUR2YXDQQUTQX0LUOJMZ01FGENVYY' 
VERSION = '20180605'

LIMIT=100
radius=800


# In[11]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[12]:


def getNearbyVenues(names, latitudes, longitudes, radius=800):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['id'],
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood',
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                    'id',
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[13]:


toronto_venues = getNearbyVenues(names=df2['Neighborhood'],
                                   latitudes=df2['Latitude'],
                                   longitudes=df2['Longitude']
                                  )


# In[14]:


#Filter out venues that are cafe or coffee shops
df_cat=toronto_venues[toronto_venues['Venue Category'].str.contains('Café|Coffee')].reset_index(drop=True)
df_cat2=toronto_venues[toronto_venues['Venue Category'].str.contains('Café')].reset_index(drop=True)
df_cat.head()


# In[15]:


# get number of coffee shops in each neighbourhood and calculate the mean
df_neigh=df_cat[['Neighborhood','Venue']].groupby('Neighborhood').count().reset_index()
df_neigh=df_neigh.merge(df2,on='Neighborhood').rename(columns={'Venue':'Coffee shops count'})
df_sort=df_neigh.sort_values(by='Coffee shops count',ascending=False)
print(df_sort.head(6))
print(df_sort['Coffee shops count'].mean())


# In[16]:


#download the json file
import json
geo_url='https://raw.githubusercontent.com/ag2816/Visualizations/master/data/Toronto2.geojson'
wget.download(geo_url,'/Users/xinrui/Downloads')

with open('/Users/xinrui/Downloads/Toronto2.geojson') as json_data:
    choro_data = json.load(json_data)


# In[17]:


#create a choropleth map showing numbers of coffee shops
map_choro = folium.Map(location=[lat, lon], zoom_start=12)

map_choro.choropleth(
    geo_data=choro_data,
    data=df_sort,
    columns=['Postal Code', 'Coffee shops count'],
    key_on='feature.properties.CFSAUID',

    fill_color='YlOrRd', 
    fill_opacity=0.4, 
    line_opacity=0.8,
    legend_name='Number of Chinese Restaurants',
    reset=True
)

# add markers to map
for lat, lng, label in zip(df_cat['Venue Latitude'], df_cat['Venue Longitude'], df_cat['Venue']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=7,
        popup=label,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        parse_html=False).add_to(map_choro)  
       

map_choro


# ## 2.1 Get restaurant information from Foursquare

# In[19]:


list_rows=[]
#get likes and ratings 
for venue_id in df_cat2['id'].tolist():
    url = 'https://api.foursquare.com/v2/venues/{}?client_id={}&client_secret={}&v={}'.format(
        venue_id, 
        CLIENT_ID, 
        CLIENT_SECRET,
        VERSION)
    getresult = requests.get(url).json()
    venue_data=getresult['response']['venue']
    venue_info=[]
    
    try:
        venue_id=venue_data['id']
        venue_name=venue_data['name']
        venue_likes=venue_data['likes']['count']
        venue_rating=venue_data['rating']
        venue_info.append([venue_id,venue_name,venue_likes,venue_rating])
    except KeyError:
        pass
        
    for row in venue_info:
        list_rows.append(venue_info)
     
    
df_info = pd.DataFrame(list_rows,columns=['Info'])
df_info.head()


# In[20]:


#split the dataframe into 4 columns
column_names=['id','Name','Likes','Rating']
df_info=pd.DataFrame(df_info['Info'].tolist(), columns=column_names)
df_info.drop_duplicates(subset='id',inplace=True)
df_info.head()


# In[21]:


df_merge=df_cat2.merge(df_info,on='id',how='right').drop_duplicates(subset='id').drop(columns='Venue').reset_index(drop=True)
df_merge.head()


# In[22]:


#Normalize the data
df_score=df_merge[['Neighborhood','Name','Likes','Rating']].merge(df_neigh,on='Neighborhood').groupby('Neighborhood').mean()
df_score.round(2)
df_score['Likes']=df_score['Likes']/df_score['Likes'].max()
df_score['Rating']=df_score['Rating']/df_score['Rating'].max()
df_score['performance']=df_score['Rating']+df_score['Likes']
df_score.reset_index(inplace=True)
df_score.head()


# In[26]:


#Plot a bar chart showing the performance
import matplotlib.pyplot as plt
top=df_score.nlargest(5,'performance')
msk1=df_score['performance']>=top.performance.min()
msk2=df_score['performance']<top.performance.min()
plt.figure(figsize=(20,8))

# colour code the bars
plt.bar(df_score['Neighborhood'][msk1],df_score['performance'][msk1],color='red',align='edge')
plt.bar(df_score['Neighborhood'][msk2],df_score['performance'][msk2],color='green',align='edge')
plt.xticks(rotation=90)
plt.title('Performance of each neighbourhood',fontsize=16)
plt.ylabel('Performance')


# # 3. K-means clustering

# In[29]:


#Normalize the data
df_score['Coffee shops count']=df_score['Coffee shops count']/df_score['Coffee shops count'].max()
df_norm=df_score[['Neighborhood','performance','Coffee shops count']]
df_norm['performance']=df_norm['performance']/df_norm['performance'].max()
df_norm.set_index('Neighborhood',inplace=True)
df_norm.head()


# In[30]:


#use k-means techinique to cluster into 5 tiers
kclusters=5
k_means = KMeans(init="k-means++", n_clusters=kclusters, n_init=12)
k_means.fit(df_norm)
labels = k_means.labels_
df_norm['tier'] = labels
df_norm.reset_index()


# In[31]:


fig, ax = plt.subplots()
ax.scatter(df_norm.performance, df_norm['Coffee shops count'],c=k_means.labels_, s=50, cmap='Accent')
ax.set(title = "Neighbourhood Clustering",
       xlabel = "Performance", 
       ylabel = "Counts")


# In[32]:


df_norm[df_norm['tier']==0]


# In[33]:


df_norm[df_norm['tier']==1]


# In[34]:


df_norm[df_norm['tier']==2]


# In[35]:


df_norm[df_norm['tier']==3]


# In[36]:


df_norm[df_norm['tier']==4]

