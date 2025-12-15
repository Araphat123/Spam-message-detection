import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample Dataset (English + Somali)
data = [
    ("Win a free iPhone now!", "spam"),
    ("Congratulations! You've won a lottery.", "spam"),
    ("Click here to claim your prize.", "spam"),
    ("Make money fast, work from home.", "spam"),
    ("Urgent! Your account is compromised.", "spam"),
    ("Hambalyo! Waxaad ku guuleysatay abaalmarin.", "spam"), # Somali: Congrats! You won a prize.
    ("Guji halkan si aad u hesho lacagtaada.", "spam"), # Somali: Click here to get your money.
    ("Lacag degdeg ah samee.", "spam"), # Somali: Make money fast.
    ("Hey, how are you doing?", "ham"),
    ("Meeting at 3 PM tomorrow.", "ham"),
    ("Can you send me the report?", "ham"),
    ("Let's grab lunch later.", "ham"),
    ("See you soon.", "ham"),
    ("Sidee tahay? Ma fiicantahay?", "ham"), # Somali: How are you? Are you good?
    ("Waan ku arki doonaa hadhow.", "ham"), # Somali: I will see you later.
    ("Ma ii soo diri kartaa warbixinta?", "ham"), # Somali: Can you send me the report?
    ("Kulanka waa berri.", "ham"), # Somali: The meeting is tomorrow.
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", "spam"),
    ("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.", "ham"),
    ("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info", "spam"),
    ("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18", "spam"),
    ("I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.", "ham"),
    ("XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL", "spam"),
    ("Oh k...i'm watching here:)", "ham"),
    ("England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+", "spam"),
    ("Is that seriously how you spell his name?", "ham"),
    ("I‘m going to try for 2 months ha ha only joking", "ham"),
    ("So ü pay first lar... Then when is da stock comin...", "ham"),
    ("Aft i finish my lunch then i go str down lor. Ard 3 smth lor. U finish ur lunch already?", "ham"),
    ("Ffffffffff. Alright no way I can meet up with you sooner?", "ham"),
    ("Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worried. He knows I'm sick when I turn down pizza. Lol", "ham"),
    ("Lol your always so convincing.", "ham"),
    ("Did you catch the bus ? Are you frying an egg ? Did you make a tea? Are you eating your mom's left over dinner ? Do you feel my Love ?", "ham"),
    ("I'm back &amp; we're packing the car now, I'll let you know if there's room", "ham"),
    ("Ahhh. Work. I vaguely remember that! What does it feel like? Lol", "ham"),
    ("Wait that's still not all that clear, were you not sure about me being sarcastic or that that's why x doesn't want to live with us", "ham"),
    ("Yeah he got in at 2 and was v apologetic. n had fallen out with his gf so was in a mood. But we're clean now so it's all good.", "ham"),
    ("K tell me anything about you.", "ham"),
    ("For fear of fainting with the of all that housework you just did? Quick have a cuppa", "ham"),
    ("Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged", "spam"),
    ("07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow", "spam"),
    ("SMS. ac Sptv: The New Jersey Devils and the Detroit Red Wings play Ice Hockey. Correct or Incorrect? End? Reply END SPTV", "spam"),
    ("Congrats! 1 year special cinema pass for 2 is yours. Call 09065064242 to claim 150p/msg", "spam"),
    ("As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589", "spam"),
    ("Urgent UR awarded a complimentary trip to EuroDisinc Trav, Aco&Entry41 Or £1000. To claim txt DIS to 87121 18+6*£1.50(more)Cls", "spam")
]

df = pd.DataFrame(data, columns=['text', 'label'])

# Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Simple stopword removal (English + Somali common words)
    stopwords = set(['the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'on', 'with', 
                     'waa', 'iyo', 'ee', 'oo', 'ku', 'ka', 'u', 'si', 'aad'])
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorization and Model Training
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Evaluation
X_test_tfidf = vectorizer.transform(X_test)
y_pred = svm_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save Models
with open('spam_app/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('spam_app/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
