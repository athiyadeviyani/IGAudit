from instagram_private_api import Client, ClientCompatPatch
from matplotlib import pyplot as plt

import getpass
import random


# INITIAL AUTHENTICATION
def login():
    username = input("username: ")
    password = getpass.getpass("password: ")
    api = Client(username, password)
    return api

api = login()

def get_ID(username):
    return api.username_info(username)['user']['pk']

def get_followers(userID, rank):
    followers = []
    next_max_id = True
    
    while next_max_id:
        if next_max_id == True: next_max_id=''
        f = api.user_followers(userID, rank, max_id=next_max_id)
        followers.extend(f.get('users', []))
        next_max_id = f.get('next_max_id', '')
    
    user_fer = [dic['username'] for dic in followers]
    
    return user_fer


# GET TARGET USER INFORMATION
username = input("Instagram username for analysis: ")

uid = get_ID(username)
rank = api.generate_uuid()


# GET USER'S FOLLOWERS
followers = get_followers(uid, rank)

print("This user has " + str(len(followers)) + " followers.")

print("============================")


# GENERATE RANDOM SAMPLE (for efficiency)
samples = int(input("Number of random samples (recommended: 1-100): "))
print("Generating " + str(samples) + " random samples for " + username + " followers!")
random_followers = random.sample(followers, samples)

print("Analyzing random samples...")


# START ANALYSIS OF THE RANDOM SAMPLE
suspicious = []
tuples = []

i = 0 
for follower in random_followers:
    f_id = get_ID(follower)
    followers = api.user_info(f_id)['user'].get('follower_count')
    followings = api.user_info(f_id)['user'].get('following_count')
    
    tuples.append((followers, followings))
    
    # SMOOTH!
    if followers == 0:
        followers = 1
        
    if followings == 0:
        followings = 1
        
    ratio = followings/followers
    
    i += 1
    
    if (i%10==0):
        print(str(i) + " out of " + str(len(random_followers)) + " followers processed.")
    
    # 'FAKENESS' THRESHOLD
    # e.g. user_x has 100 followers and 2000 followings, user_x is flagged 'suspicious'
    if ratio > 20:
        suspicious.append(follower)
        
print(str(len(suspicious)) + " suspicious accounts detected!")


# CALCULATE THE OVERALL AUTHENTICITY OF THE USER
percentage_fake = len(suspicious)*100/len(random_followers)

print(username + " has " + str(100-percentage_fake) + "% authenticity!")


# GENERATE THE GRAPH
x = [x[0] for x in tuples]
y = [x[1] for x in tuples]

f, ax = plt.subplots(figsize=(16,10))
plt.scatter(x,y)
plt.plot([i for i in range(max(max(x), max(y)))], 
         [i for i in range(max(max(x), max(y)))], 
         color = 'red', 
         linewidth = 2, label='following:followers = 1:1'
         )
plt.text(2500, 4000, 'following:followers = 1:1', size=14)
plt.title('Following:Followers plot for user:' + username + ' Instagram Followers', size=20)
plt.xlabel('Followers', size=14)
plt.ylabel('Following', size=14)

plt.show()
