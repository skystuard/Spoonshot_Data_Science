#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries 

# In[132]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re


# ## Inserting Data

# In[133]:


ingredients= ['Ambrette Seed',
'Apple Cinnamon Granola',
'Arizona Seasoning',
'Americano Coffee',
'Baby Abalone',
'Cadbury Double Decker Chocolate Bar',
'Campari Tomato',
'Celery Soup',
'Chia Meal',
'Crunch Bars',
'Cardamom',
'Giardiniera',
'Hog Maw',
'Mccormick Montreal Steak Seasoning',
'Muesli',
'Mulberry',
'Munch Chocolate',
'Murukku Packet',
'Mango',
'Organic Maize',
'Organic Peruvian Groundcherry',
'Organic Tartar Cream',
'Orange Extract',
'Pickled Cauliflower',
'Pork Chump Chops',
'Pork Lungs',
'Pork Tripe',
'Peanut Butter',
'Smokies Sausage',
'Snickers Spread',
'Strawberry Gelatin',
'Salmon'
'Tomato',
'Tamarind',
'Vegan Carob Chips',
'Vegan Chicken Strips',
'Vegan Chorizo',
'Vegan Marshmallow',
'Vegan Puff Pastry Sheet',
'Vegan Semisweet Chocolate Chips',
'Vegan White Cake',
'Vegetable Stock',
'Vinegar']


# In[134]:


article = """munchbox meals munchbox is a geelong-based boutique meal prep and delivery service,specialising in artisan vegan and vegetarian meals and munchies. with a passion forproviding delicious and nutritious food that is ethical, sustainable and affordable, allmunchbox meals are made from scratch using a balance of the freshest local and organicvegetables and fruit, grains, seeds and nuts and homegrown herbs. we chat to eden callick –the brains behind the business – about all things munchbox. hey! thanks for chatting to forte!first up, can you explain how munchbox became to be? munchbox was a daydream turnedreality. i was in a really draining job i hated, and would spend the majority of my lunch-breaklooking for vegan food. the struggle to find a tasty, good quality, vegan meal is real.sometimes it was so real i’d just give up looking, and go hungry. one day i hungrily andjokingly announced, ‘i’m going to open my own vegan food business’ and with that impulsiveannouncement, i quit my job almost immediately, and promised myself i would put my wholeheart and soul in to becoming the best vegan chef i can be. with a whole lot of determination,passion and support from those around me, munchbox was born and is now my baby! iguess a little bit of selfishness for me to have passion and purpose in life, as well as the lackof real quality vegan cuisine on the geelong food scene is really how munchbox came to be– out of necessity. where did your passion for vegan/vegetarian food come from? i’ve alwayshad a love and passion for food, but it wasn’t until i slowly transitioned to vegan that mypassion turned into a true obsession! it opened a whole new door for experimentation in thekitchen; i want to create cruelty free food that tastes and looks better, but most importantly isbetter for you! nothing makes me happier then feeding people and changing their views onvegan food. can you give our readers a run-down of how ‘munchbox’ works? munchbox isan artisan meal prep and delivery service, providing chef made meals to you. our muncherscan order online any time before friday midnight, with meals being ready for pick-up ordelivery on sunday. ready for the week ahead! these hours will soon be changing to include24/7 online ordering, with pick-up and delivery between 7am-7pm seven days a week. allmeals come ready to eat or heat; those that are best enjoyed heated include a lit flame! wealso specialise in events and corporate catering; so, if there’s a bunch that’s got the munch,we’re here to feed your sweet souls! where have you gathered most of your recipes andideas from? i’m constantly gathering ideas and inspiration from anywhere and everywhere! ihave a collection of paper scraps with scribbled recipes, and sleep with a notebook next tomy bed just in case i dream up a new idea. i have mood-boards, and future menu lists. it’s amess really! i find so much inspiration and ideas from traditional non-vegan meals, and havebeen fortunate enough to work alongside some amazing non-vegan chefs who have (taught)and inspired me incredibly. my mumma and sister, who are huge vegan foodies, are alwayscoming up with insane recipes for me as well. but each recipe is ultimately my own, madefrom scratch, tried and tested until it’s perfected. how do you determine what local andorganic ingredients make the cut? we are passionately committed to sustainability andsupporting our local community. our menus are always based around the freshest seasonalproduce, sourced from local organic farmers, as well as other artisan vegan businesses whosupply some of our ingredients. we also lovingly grow all our own herbs, and as munchboxexpands so will our veggie garden; to include some pretty crazy exotic fruit and vege! whattype of people does this service suit? munchbox is perfect for those who are time poor, buthungry, and care what they put in to their body. looking for a delicious and nutritious meal,that isn’t frozen or full of crap. we put all the freshness, flavour, love and care into each meal,so you don’t have to! breaky, lunch, or tea, pick up or delivery, we’ve got you covered so youdon’t have to think about cooking a thing! munchbox isn’t exclusively for vegetarians orvegans, but is perfect for those looking to try a tasty healthy alternative! do you do customorders? or is everything off the set menu? yes! all our meals are fully customisable to suitdiet and taste! don’t like tomato? just let us know! gluten intolerant? we got you! you can addor remove any ingredients and extras in our meals when you order. however, if you’d like ourchefs to create something extra special just for you, we’re more than happy to! we cancustomise personalised menus, meal plans and munchboxes just for you! where are youhoping to take munchbox in future? munchbox has big plans for the future! you can expectan ever-changing menu (to keep up with my ever-changing mind). we’ll be incorporatingmenulog and uber eats as apart of our service, and expanding trading hours and servicelocations in the very near future. there may also be plans for ‘munch mobile’ to attendfestivals, markets, and events.. but i can’t give too much away! we have sooo manysurprises for ya’ll, but i can’t spoil them all at once! check it out via insta @vegan.munchboxor you can get your munch on at munchboxmeals.com.au"""


# ## cleaning the data

# In[135]:


article=re.sub('[^a-zA-Z]',' ', article)
article=article.lower()


# In[136]:


article


# ## Tokenizing

# In[137]:


token=nltk.word_tokenize(article)
token


# In[138]:


stopwords.words('english')


# ## Lemmitizing

# In[139]:


lemmatizer=WordNetLemmatizer()
tok = []
for i in range (len(token)):
    tok.append([word for word in token if word not in stopwords.words('english')])
tok


# ## Reshaping the token in the required dimensions 

# In[140]:


tok = tuple(lemmatizer.lemmatize(i) for i in tok[0])
for i in range (len(tok)):
    tok = [word for word in tok]
tok = np.array(tok).reshape(1, 485).tolist() 
tok


# ## Importing and Implementing TF-IDF

# In[141]:


from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(tok):
    return tok

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    
    token_pattern=None)  


# In[142]:


tfidf.fit(tok)
dic = tfidf.vocabulary_
dic


# ## Summing Frequency of items of ingredients found in the dic and ranking according 

# In[155]:


d={}
for items in ingredients:
    sum = 0
    lis = items.split()
    for item in lis:
        if dic.get(item.lower()):
            sum = sum +  dic[item.lower()]
    
    d[items] = sum
d


# In[147]:


print(dic.get('seed'))


# ## Arranging maximum rank of list in decsending order

# In[162]:


# final list
l = {k:v for k, v in sorted(d.items(), key= lambda item:item[1],reverse=True)}
l


# In[ ]:




