"""
spam_detector.py  —  Spam / Ham / Smishing Detector
====================================================
HOW TO RUN:   python spam_detector.py

First run: trains models automatically (~2 min), then opens GUI.
Next runs:  loads saved models instantly and opens GUI.

If predictions seem wrong, delete saved_models_final/ and rerun.
"""

import warnings; warnings.filterwarnings('ignore')
import os, sys, re, pickle, threading, time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import scipy.sparse as sp

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'saved_models_final')
DATA_PATH   = os.path.join(SCRIPT_DIR,
    'A Balanced Dataset for Spam and Smishing Detection', 'Dataset_10191.csv')

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (shared by training and GUI prediction)
# ═══════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    'i','me','my','myself','we','our','ours','you','your','yours','yourself',
    'he','him','his','himself','she','her','hers','herself','it','its','itself',
    'they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by',
    'for','with','about','against','between','into','through','during','before',
    'after','above','below','to','from','up','down','in','out','on','off',
    'over','under','again','further','then','once','here','there','when',
    'where','why','how','all','both','each','few','more','most','other','some',
    'such','no','nor','not','only','own','same','so','than','too','very',
    'can','will','just','should','now','s','t','d','ll','m','re','ve'
}

def _stem(w):
    for suf in ['tion','tions','ing','ment','ness','ful','less','ous','ive',
                'ed','ly','er','al']:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    if w.endswith('ies') and len(w) > 4: return w[:-3] + 'i'
    if w.endswith('es')  and len(w) > 3: return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and len(w) > 3: return w[:-1]
    return w

def preprocess(text):
    t = str(text).lower()
    t = re.sub(r'http\S+|www\.\S+',              ' urllink ',     t)
    t = re.sub(r'\S+@\S+',                        ' emailaddr ',   t)
    t = re.sub(r'\b\d[\d\s\-\.()]{6,}\d\b',      ' phonenumber ', t)
    t = re.sub(r'[^\w\s]',                        ' ',             t)
    t = re.sub(r'\b\d+\b',                        ' numbr ',       t)
    return ' '.join(_stem(w) for w in t.split()
                    if w not in STOPWORDS and len(w) > 1)

# ── Keyword lists (aligned with actual dataset labels) ────────────────────
# SPAM  = commercial SMS: ringtones, subscriptions, telecoms, adult, retail
# SMISHING = prize/lottery fraud OR bank/account fraud (both in this dataset)

_SPAM_COMMERCIAL = [
    'ringtone','tone','nokia','polyphonic','wallpaper','screensaver',
    'subscribe','subscription','per week','per month','txt stop','stop2stop',
    'orange','o2 mobile','airtel','mobileupd8','mobile bundle',
    'sexy','adult','cam moby','fantasy chat','lyricalladie',
    'shop now','retail','click to shop','special deal',
    'work from home','make money','earn daily','extra cash',
    'passive income','investment opportunity','unlimited income',
    'secret method','easy money','making money','earn money',
    'double your','not a scam','i promise it','100% legit',
]
_SMISHING_PRIZE = [
    'you have won','you won','winner','awarded','selected to receive',
    'prize guaranteed','cash prize','prize worth','claim your prize',
    'prize from','guaranteed prize','lucky draw','lucky winner',
    'complimentary','holiday or £','tenerife','ibiza','city break',
    'gift voucher','gift card','£500','£1000','£2000','£5000',
    '£10,000','£200','£350','£900','rs.2,00,000','inr',
]
_SMISHING_BANK = [
    'suspended','blocked','verify','confirm','unauthorized','locked',
    'expire','validate','action required','security alert','login attempt',
    'update your','restore access','compromised','credential',
    'account has been','unusual activity','your card','your account',
    'click to verify','identity check','re-verify','kyc','paytm',
    'reactivate your','apple id','icloud','your bank',
]
_HAM_KW = [
    'bro','dude','mate','lol','haha','omg','btw','imo',
    'see you','talk later','catch up','hang out','going out',
    'lunch','dinner','coffee','party','birthday',
    'school','class','lecture','exam','assignment',
    'love you','miss you','take care','get well',
    'passed the exam','got the job','great news',
]

def extract_features(text):
    """Returns a fixed-length list of hand-crafted features."""
    t   = str(text).lower()
    raw = str(text)

    # ── binary signals ─────────────────────────────────────────────────────
    has_url      = int(bool(re.search(r'http\S+|www\.\S+|bit\.ly|tinyurl', t)))
    has_email    = int(bool(re.search(r'\S+@\S+', t)))
    # Phone number — key smishing signal
    has_phone    = int(bool(re.search(r'\b0[78]\d{8,9}\b|\b\d{10,11}\b|\+\d{10,12}', raw)))
    has_shortcode = int(bool(re.search(r'\btxt\b.*\b\d{4,6}\b|\b\d{4,6}\b.*\btxt\b|'
                                       r'text.*\b\d{5}\b|\b\d{5}\b.*text', t)))
    # Brand names (smishing bank type)
    has_bank_brand = int(any(b in t for b in [
        'paypal','barclays','hsbc','lloyds','natwest','halifax',
        'santander','amazon','apple','microsoft','dvla','hmrc','irs',
        'vodafone','paytm','flipkart','cibc','abta']))
    # Keyword groups
    has_spam_comm  = int(any(k in t for k in _SPAM_COMMERCIAL))
    has_smish_prize= int(any(k in t for k in _SMISHING_PRIZE))
    has_smish_bank = int(any(k in t for k in _SMISHING_BANK))
    has_ham        = int(any(k in t for k in _HAM_KW))

    # ── interaction features ────────────────────────────────────────────────
    # Prize + phone number = smishing (not spam)
    prize_and_phone  = has_smish_prize * has_phone
    # Bank brand + url/phone = smishing
    bank_and_contact = has_bank_brand * (has_url + has_phone)
    # Commercial offer + shortcode = spam
    comm_and_code    = has_spam_comm * has_shortcode
    # Prize but no phone = more like spam
    prize_no_phone   = has_smish_prize * (1 - has_phone)
    # Spam commercial without prize keywords
    pure_spam_comm   = has_spam_comm * (1 - has_smish_prize)
    # Ham signal with no suspicious keywords
    pure_ham         = has_ham * (1 - has_smish_prize) * (1 - has_smish_bank) * (1 - has_spam_comm)

    # ── style signals ───────────────────────────────────────────────────────
    urgency    = min(sum(1 for k in ['urgent','immediately','asap',
                                     'act now','today only','right now',
                                     'limited time','expires soon',
                                     'last chance','final'] if k in t), 3)
    caps_ratio  = sum(1 for c in raw if c.isupper()) / max(len(raw), 1)
    excl_count  = min(raw.count('!'), 5)
    word_count  = len(raw.split())
    has_gbp     = int('£' in raw)
    has_rupee   = int('rs.' in t or 'inr' in t or '₹' in raw)
    has_shorturl= int(bool(re.search(r'bit\.ly|tinyurl|goo\.gl|t\.co', t)))
    link_urgency= has_url * urgency
    phone_urgency= has_phone * urgency
    casual_tone = int(bool(re.search(
        r'\bbro\b|\bdude\b|\bmate\b|\blol\b|\bhaha\b|\bomg\b|\bbtw\b', t)))

    return [
        # raw signals (9)
        has_url, has_email, has_phone, has_shortcode, has_bank_brand,
        has_spam_comm, has_smish_prize, has_smish_bank, has_ham,
        # interaction features (6)
        prize_and_phone, bank_and_contact, comm_and_code,
        prize_no_phone, pure_spam_comm, pure_ham,
        # style (10)
        urgency, caps_ratio, excl_count, word_count,
        has_gbp, has_rupee, has_shorturl,
        link_urgency, phone_urgency, casual_tone,
    ]  # total: 25 features

HARD_SAMPLES = [
    # ── HAM that looks like spam/smishing ─────────────────────────────────
    ("Congrats bro you passed the exam!", "ham"),
    ("Free time today? Lets go out", "ham"),
    ("You wont believe what happened today lol", "ham"),
    ("Congratulations on getting the job!", "ham"),
    ("You won the match! Amazing game bro", "ham"),
    ("Did you get the bonus this month at work?", "ham"),
    ("Bro you wont believe what she said lol", "ham"),
    ("Free lunch today at the office!", "ham"),
    ("Cash or card? Let me know before we go out", "ham"),
    ("I got a new job offer, huge pay raise!", "ham"),
    ("Buy me a coffee when youre free bro?", "ham"),
    ("I cant believe you won the tournament!", "ham"),
    ("They gave everyone a cash bonus at work today!", "ham"),
    ("Dude I won the school raffle haha", "ham"),
    ("Hey are you free this evening? Lets catch up", "ham"),
    ("Omg I won the debate competition lol", "ham"),
    ("We won the football match yesterday bro!", "ham"),
    ("Claim your free seat at the event tonight", "ham"),

    # ── TRUE SPAM (commercial SMS — ringtones, subscriptions, mobile offers)
    ("Free ringtone! Text NOKIA to 87021. 1st tone free!", "spam"),
    ("Double Mins & Double Txt on latest Orange mobiles. Call MobileUpd8 on 0800", "spam"),
    ("Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121", "spam"),
    ("8007 FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007", "spam"),
    ("Sunshine Quiz! Win a Sony DVD player. Txt QUIZ to 87575", "spam"),
    ("U are subscribed to Mobile Content Service for £3 per 10 days. Send STOP to cancel", "spam"),
    ("Jamster! Get your free wallpaper text HEART to 88888 now!", "spam"),
    ("Hot Live Fantasies call now 08707509020 Just 20p per min", "spam"),
    ("LYRICALLADIE(21/F) wants to be your friend. Reply YES or NO", "spam"),
    ("Get the official England ringtone for tonights game! text TONE to 83600", "spam"),
    ("Non-stop internet with Pocket Internet. Dial *234*2900# now", "spam"),
    ("Fantasy football is back! Go to Sky Gamestar. Win £250k dream team", "spam"),
    ("FreeMsg: Feelin kinda lonely hope u like 2 keep me company! Got a cam moby", "spam"),
    ("Txt to WIN! Free 1st week entry 2 TEXTPOD 4 a chance 2 win 40GB iPod. Txt POD to 84128", "spam"),
    ("Dear subscriber ur draw 4 £100 gift voucher will b entered on receipt of correct ans", "spam"),
    ("PRIVATE! Your account shows 800 un-redeemed points. Call 08718738002 Identifier Code:", "spam"),

    # ── TRUE SMISHING (prize/lottery fraud — has phone number to call)
    ("WINNER!! You've won £5000! Call 07123456789 NOW to claim your prize.", "smishing"),
    ("WINNER you have won 5000 Call 09061749602 NOW to claim", "smishing"),
    ("URGENT! Your Mobile number has been awarded with a £2000 prize. Call 09058094455 now", "smishing"),
    ("Congratulations! You have won a £1000 cash prize! Call 09061701461 to claim now", "smishing"),
    ("Todays Vodafone numbers ending 5263 are selected to receive Rs.2,00,000 award. Call now", "smishing"),
    ("You have WON a guaranteed £1000 cash or a £2000 prize. Call 09050000327 to claim", "smishing"),
    ("URGENT! We are trying to contact U. Your £900 prize is still awaiting collection. Call 09071517866", "smishing"),
    ("complimentary 4 STAR Ibiza Holiday or £10,000 cash needs your URGENT collection. Call 09066364349", "smishing"),
    ("URGENT! Your Mobile number has been awarded INR.2,00,000 prize GUARANTEED. Call 7908807538", "smishing"),
    ("Win a £1000 cash prize or a prize worth £5000. Call 09050000460", "smishing"),
    ("You are selected to receive a £350 award. Call 09061743886 from landline now", "smishing"),
    ("FREE entry: txt WIN to 80085 to claim your £200 prize. Call 0906 170 0461", "smishing"),

    # ── TRUE SMISHING (bank/account fraud)
    ("Your bank account has been suspended, call us immediately on 08001561911", "smishing"),
    ("HSBC: Unusual activity detected. Verify your identity now at http://hsbc-secure.com", "smishing"),
    ("Your PayPal account will be closed unless you confirm details at http://paypal-verify.net", "smishing"),
    ("Amazon: Your order is on hold, update payment at http://amz-verify.com", "smishing"),
    ("Your NatWest card has been blocked due to suspicious login. Call 08457346090", "smishing"),
    ("IRS: Final tax notice. Respond immediately or face legal action. Call 08001234567", "smishing"),
    ("Your Apple ID is locked. Verify at http://apple-support-id.net", "smishing"),
    ("Dear customer your account has been compromised, verify now at http://secure-bank.net", "smishing"),
    ("Santander security alert: unusual login detected, click to verify http://santander-alert.com", "smishing"),

    # ── SPAM that sounds like social engineering (no phone number)
    ("This is not a scam I promise you can earn daily from this site", "spam"),
    ("Hey I found a way to make extra cash online, want the link?", "spam"),
    ("A friend started doing this and makes 500 a week from home", "spam"),
    ("Just wanted to share this investment opportunity with you", "spam"),
    ("Work from home and earn unlimited income daily", "spam"),
    ("Secret method that millionaires dont want you to know", "spam"),
    ("You can make passive income with zero effort guaranteed", "spam"),
    ("Click this link to double your salary this month", "spam"),
    ("Earn unlimited income from home, no experience needed", "spam"),
    ("It is 100% legit, I promise it works, easy money daily", "spam"),
]


def train_models(log_fn=print):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    log_fn("[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    hard = pd.DataFrame(HARD_SAMPLES, columns=['TEXT', 'LABEL'])
    df = pd.concat([df, hard], ignore_index=True)
    log_fn(f"      {len(df)} total rows  |  {df['LABEL'].value_counts().to_dict()}")

    log_fn("[2/5] Preprocessing text...")
    df['clean'] = df['TEXT'].apply(preprocess)

    log_fn("[3/5] Building feature matrix...")
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=6000,
                            sublinear_tf=True, min_df=1)
    X_tfidf = tfidf.fit_transform(df['clean'])
    X_hand  = np.array([extract_features(t) for t in df['TEXT']], dtype=float)

    # Scale hand features so they're not dwarfed by TF-IDF
    from sklearn.preprocessing import MaxAbsScaler
    scaler  = MaxAbsScaler()
    X_hand_scaled = scaler.fit_transform(X_hand)
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_hand_scaled)], format='csr')

    le = LabelEncoder()
    y  = le.fit_transform(df['LABEL'])
    log_fn(f"      Feature matrix: {X.shape}  |  Classes: {list(le.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    log_fn("[4/5] Training models (this takes ~2 minutes)...")
    trained = {}

    # Logistic Regression
    log_fn("      → Logistic Regression...")
    lr = LogisticRegression(C=5, max_iter=2000, class_weight='balanced',
                            solver='lbfgs', multi_class='auto', random_state=42)
    lr.fit(X_tr, y_tr)
    r = classification_report(y_te, lr.predict(X_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_logistic_regression'] = lr

    # Random Forest
    log_fn("      → Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r = classification_report(y_te, rf.predict(X_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_random_forest'] = rf

    # SVM
    log_fn("      → SVM...")
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=3000, class_weight='balanced', random_state=42))
    svm.fit(X_tr, y_tr)
    r = classification_report(y_te, svm.predict(X_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_svm'] = svm

    # Naive Bayes (needs non-negative input — use TF-IDF only)
    log_fn("      → Naive Bayes...")
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    nb = MultinomialNB(alpha=0.1)
    nb.fit(Xn_tr, yn_tr)
    r = classification_report(yn_te, nb.predict(Xn_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_naive_bayes'] = nb

    # Decision Tree
    log_fn("      → Decision Tree...")
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=20, class_weight='balanced', random_state=42)
    dt.fit(X_tr, y_tr)
    r = classification_report(y_te, dt.predict(X_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_decision_tree'] = dt

    # K-Nearest Neighbors
    log_fn("      → KNN...")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=7, metric='cosine', n_jobs=-1)
    knn.fit(X_tr, y_tr)
    r = classification_report(y_te, knn.predict(X_te),
                               target_names=le.classes_, output_dict=True)
    log_fn(f"         Accuracy: {r['accuracy']*100:.1f}%  "
           f"| ham:{r['ham']['f1-score']*100:.0f}%  "
           f"spam:{r['spam']['f1-score']*100:.0f}%  "
           f"smishing:{r['smishing']['f1-score']*100:.0f}%")
    trained['model_knn'] = knn

    # Validate on tricky cases
    log_fn("\n[4b] Tricky case validation:")
    tricky = [
        ("Congrats bro you passed the exam!", "ham"),
        ("Free time today? Lets go out", "ham"),
        ("You wont believe what happened today lol", "ham"),
        ("WINNER!! Youve won £5000! Call NOW to claim your prize.", "spam"),
        ("WINNER you have won 5000 Call NOW to claim", "spam"),
        ("This is not a scam I promise you can earn daily from this site", "spam"),
        ("Work from home earn unlimited income daily", "spam"),
        ("Your Barclays account is locked verify at http://barclays-secure.com", "smishing"),
        ("Your Apple ID locked verify now http://apple-id-check.net", "smishing"),
        ("Hey are you free this evening lets catch up", "ham"),
        ("We won the football match yesterday bro", "ham"),
        ("Earn unlimited income from home, no experience needed", "spam"),
    ]

    def _vec(text):
        h = np.array([extract_features(text)], dtype=float)
        h_scaled = scaler.transform(h)
        return sp.hstack([tfidf.transform([preprocess(text)]),
                          sp.csr_matrix(h_scaled)], format='csr')

    ok = 0
    for msg, true in tricky:
        pred = le.inverse_transform(lr.predict(_vec(msg)))[0]
        mark = 'OK   ' if pred == true else 'WRONG'
        if pred == true: ok += 1
        log_fn(f"      {mark} | true={true:<10} pred={pred:<10} | {msg[:55]}")
    log_fn(f"      Score: {ok}/{len(tricky)}")

    log_fn("\n[5/5] Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(tfidf,  open(f'{MODEL_DIR}/tfidf.pkl',   'wb'), protocol=4)
    pickle.dump(le,     open(f'{MODEL_DIR}/le.pkl',      'wb'), protocol=4)
    pickle.dump(scaler, open(f'{MODEL_DIR}/scaler.pkl',  'wb'), protocol=4)
    for name, m in trained.items():
        pickle.dump(m, open(f'{MODEL_DIR}/{name}.pkl', 'wb'), protocol=4)
    log_fn(f"      Saved to: {MODEL_DIR}")
    log_fn("Done! Models ready.")
    return tfidf, le, scaler, trained


def load_models():
    tfidf  = pickle.load(open(f'{MODEL_DIR}/tfidf.pkl',  'rb'))
    le     = pickle.load(open(f'{MODEL_DIR}/le.pkl',     'rb'))
    scaler = pickle.load(open(f'{MODEL_DIR}/scaler.pkl', 'rb'))
    models = {}
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith('model_') and fname.endswith('.pkl'):
            name = fname.replace('model_', '').replace('.pkl', '').replace('_', ' ').title()
            models[name] = pickle.load(open(f'{MODEL_DIR}/{fname}', 'rb'))
    return tfidf, le, scaler, models


def models_exist():
    if not os.path.isdir(MODEL_DIR):
        return False
    files = os.listdir(MODEL_DIR)
    return (os.path.exists(f'{MODEL_DIR}/tfidf.pkl') and
            os.path.exists(f'{MODEL_DIR}/le.pkl') and
            os.path.exists(f'{MODEL_DIR}/scaler.pkl') and
            any(f.startswith('model_') for f in files))


def classify_message(message, tfidf, le, scaler, models):
    clean  = preprocess(message)
    h      = np.array([extract_features(message)], dtype=float)
    h_sc   = scaler.transform(h)
    v_tf   = tfidf.transform([clean])

    results = {}
    for name, model in models.items():
        # NB uses only TF-IDF; others use full vector
        if 'Naive' in name or 'Nb' in name:
            vec = v_tf
        else:
            vec = sp.hstack([v_tf, sp.csr_matrix(h_sc)], format='csr')

        # Handle any feature count mismatch gracefully
        exp = getattr(model, 'n_features_in_', None)
        if exp and vec.shape[1] != exp:
            if vec.shape[1] < exp:
                pad = sp.csr_matrix((1, exp - vec.shape[1]))
                vec = sp.hstack([vec, pad], format='csr')
            else:
                vec = vec[:, :exp]

        pred  = model.predict(vec)[0]
        prob  = model.predict_proba(vec)[0]
        label = le.inverse_transform([pred])[0].upper()
        results[name] = {
            'label':      label,
            'confidence': round(max(prob) * 100, 1),
            'probs':      {le.classes_[i]: round(p * 100, 1)
                           for i, p in enumerate(prob)},
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════

BG      = '#0F1117'
PANEL   = '#1A1D27'
ACCENT  = '#6C63FF'
HAM_C   = '#1D9E75'
SPAM_C  = '#E24B4A'
SMISH_C = '#9B8EF5'
TEXT_C  = '#E8E8F0'
MUTED   = '#6B6B8A'
BORDER  = '#2A2D3E'
LABEL_COLORS = {'HAM': HAM_C, 'SPAM': SPAM_C, 'SMISHING': SMISH_C}
LABEL_ICONS  = {'HAM': '✅', 'SPAM': '🔴', 'SMISHING': '⚠️'}

SAMPLE_MESSAGES = [
    ("✅ Ham",        "Hey bro, are you free this evening? Let's catch up!"),
    ("🔴 Spam",      "8007 FREE for 1st week! No1 Nokia tone 4 ur mob. Txt NOKIA to 87021 now!"),
    ("⚠️ Prize Smish","WINNER!! You've won £5000! Call 07123456789 NOW to claim your prize."),
    ("⚠️ Bank Smish", "URGENT: Your Barclays account is locked. Verify at http://barclays-secure-login.com"),
    ("🧠 Tricky",    "Work from home and earn unlimited income daily. Message me for the link!"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Spam · Ham · Smishing Detector')
        self.geometry('1020x800')
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(820, 600)

        self.tfidf  = None
        self.le     = None
        self.scaler = None
        self.models = None
        self.ready  = False

        self._build_ui()
        threading.Thread(target=self._init_models, daemon=True).start()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, height=56)
        hdr.pack(fill='x'); hdr.pack_propagate(False)
        tk.Label(hdr, text='📧  Spam · Ham · Smishing Detector',
                 font=('Helvetica', 16, 'bold'), fg='white', bg=ACCENT
                 ).pack(side='left', padx=20)
        self.status_var = tk.StringVar(value='⏳  Initialising…')
        tk.Label(hdr, textvariable=self.status_var,
                 font=('Helvetica', 9), fg='#CCCCFF', bg=ACCENT
                 ).pack(side='right', padx=18)

        # Training log (shown only during training)
        self.log_frame = tk.Frame(self, bg=PANEL, padx=12, pady=8)
        self.log_text  = scrolledtext.ScrolledText(
            self.log_frame, height=14, font=('Consolas', 9),
            bg='#0A0D14', fg='#A0FFA0', state='disabled',
            relief='flat', bd=0)
        self.log_text.pack(fill='both', expand=True)
        self.log_frame.pack(fill='both', expand=True)

        # Main area (hidden until ready)
        self.main_frame = tk.Frame(self, bg=BG)

        # Input
        inp = tk.Frame(self.main_frame, bg=BG, pady=10, padx=16)
        inp.pack(fill='x')
        tk.Label(inp, text='Enter a message to classify:',
                 font=('Helvetica', 11, 'bold'), fg=TEXT_C, bg=BG
                 ).pack(anchor='w')
        self.txt = scrolledtext.ScrolledText(
            inp, height=5, font=('Consolas', 11),
            bg=PANEL, fg=TEXT_C, insertbackground=TEXT_C,
            relief='flat', bd=0, wrap='word',
            highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground=BORDER)
        self.txt.pack(fill='x', pady=(6, 0))
        self.txt.insert('1.0', 'Type or paste a message here…')
        self.txt.config(fg=MUTED)
        self.txt.bind('<FocusIn>',  self._ph_clear)
        self.txt.bind('<FocusOut>', self._ph_restore)

        # Buttons row
        br = tk.Frame(self.main_frame, bg=BG, pady=6, padx=16)
        br.pack(fill='x')
        self.btn = tk.Button(br, text='🔍  Classify',
                             font=('Helvetica', 11, 'bold'),
                             bg=ACCENT, fg='white',
                             activebackground='#5A52E0',
                             relief='flat', padx=20, pady=8,
                             cursor='hand2', command=self._classify)
        self.btn.pack(side='left', padx=(0, 10))
        tk.Button(br, text='🗑  Clear',
                  font=('Helvetica', 10), bg=PANEL, fg=TEXT_C,
                  activebackground=BORDER, relief='flat',
                  padx=14, pady=8, cursor='hand2',
                  command=self._clear).pack(side='left', padx=(0, 20))

        tk.Label(br, text='Try sample:',
                 font=('Helvetica', 9), fg=MUTED, bg=BG
                 ).pack(side='left', padx=(0, 6))
        colors = [HAM_C, SPAM_C, SMISH_C, '#B8860B', '#B8860B']
        for (lbl, msg), col in zip(SAMPLE_MESSAGES, colors):
            tk.Button(br, text=lbl, font=('Helvetica', 8, 'bold'),
                      bg=col, fg='white', relief='flat',
                      padx=8, pady=4, cursor='hand2',
                      command=lambda m=msg: self._load(m)
                      ).pack(side='left', padx=2)

        tk.Frame(self.main_frame, bg=BORDER, height=1).pack(fill='x', padx=16)

        # Verdict banner
        self.vf = tk.Frame(self.main_frame, bg=BG, pady=8, padx=16)
        self.vf.pack(fill='x')
        self.vlbl = tk.Label(self.vf,
                             text='Result will appear here.',
                             font=('Helvetica', 15, 'bold'),
                             fg=MUTED, bg=BG)
        self.vlbl.pack()

        # Progress bar
        self.prog = ttk.Progressbar(self.main_frame, mode='indeterminate', length=300)

        # Results table
        outer = tk.Frame(self.main_frame, bg=BG, padx=16, pady=4)
        outer.pack(fill='both', expand=True)
        cols = ('Algorithm', 'Prediction', 'Confidence', 'Ham %', 'Spam %', 'Smishing %')
        self.tree = ttk.Treeview(outer, columns=cols, show='headings', height=10)
        style = ttk.Style(); style.theme_use('clam')
        style.configure('Treeview',
                         background=PANEL, foreground=TEXT_C,
                         fieldbackground=PANEL, rowheight=30,
                         font=('Helvetica', 10))
        style.configure('Treeview.Heading',
                         background=BORDER, foreground=TEXT_C,
                         font=('Helvetica', 10, 'bold'), relief='flat')
        style.map('Treeview', background=[('selected', ACCENT)])
        for col, w in zip(cols, [200, 120, 110, 80, 80, 110]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor='center')
        sb = ttk.Scrollbar(outer, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tree.tag_configure('HAM',      background='#0D2E22', foreground=HAM_C)
        self.tree.tag_configure('SPAM',     background='#2E0D0D', foreground=SPAM_C)
        self.tree.tag_configure('SMISHING', background='#1E1830', foreground=SMISH_C)

        # Debug bar
        dbf = tk.Frame(self.main_frame, bg=PANEL, padx=14, pady=4)
        dbf.pack(fill='x', side='bottom')
        self.dblbl = tk.Label(dbf, text='', font=('Consolas', 8),
                              fg=MUTED, bg=PANEL, anchor='w')
        self.dblbl.pack(fill='x')

        # Footer
        ft = tk.Frame(self.main_frame, bg=PANEL, height=28)
        ft.pack(fill='x', side='bottom'); ft.pack_propagate(False)
        tk.Label(ft, text='NLP Project · Spring 2025–2026 · Dr. Wafaa Samy  |  '
                          'Rebuilt from scratch · sklearn-version-safe',
                 font=('Helvetica', 8), fg=MUTED, bg=PANEL).pack(pady=5)

    # ── Model init ─────────────────────────────────────────────────────────
    def _log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')
        self.update_idletasks()

    def _init_models(self):
        try:
            need_train = not (
                os.path.isdir(MODEL_DIR) and
                os.path.exists(f'{MODEL_DIR}/tfidf.pkl') and
                os.path.exists(f'{MODEL_DIR}/le.pkl') and
                os.path.exists(f'{MODEL_DIR}/scaler.pkl') and
                any(f.startswith('model_')
                    for f in os.listdir(MODEL_DIR))
            )
            if need_train:
                self.after(0, lambda: self.status_var.set('🔧  Training models — please wait…'))
                self.after(0, lambda: self._log(
                    'No trained models found. Training now...\n'
                    'This takes about 2 minutes. Please wait.\n' + '─'*60))
                tfidf, le, scaler, trained = train_models(log_fn=lambda m: self.after(0, lambda m=m: self._log(m)))
                self.tfidf  = tfidf
                self.le     = le
                self.scaler = scaler
                self.models = {k.replace('model_','').replace('_',' ').title(): v
                               for k, v in trained.items()}
            else:
                self.after(0, lambda: self.status_var.set('📂  Loading saved models…'))
                self.after(0, lambda: self._log('Loading saved models...'))
                self.tfidf, self.le, self.scaler, self.models = load_models()
                self.after(0, lambda: self._log('Models loaded successfully!'))

            n = len(self.models)
            self.ready = True
            self.after(0, lambda: self.status_var.set(f'✅  {n} models ready'))
            self.after(0, self._show_main)
        except Exception as ex:
            err = str(ex)
            self.after(0, lambda: self._log(f'\nERROR: {err}'))
            self.after(0, lambda: self.status_var.set('❌  Error — see log above'))

    def _show_main(self):
        self.log_frame.pack_forget()
        self.main_frame.pack(fill='both', expand=True)

    # ── Input helpers ──────────────────────────────────────────────────────
    def _ph_clear(self, _=None):
        if self.txt.get('1.0', 'end-1c') == 'Type or paste a message here…':
            self.txt.delete('1.0', 'end')
            self.txt.config(fg=TEXT_C)

    def _ph_restore(self, _=None):
        if not self.txt.get('1.0', 'end-1c').strip():
            self.txt.insert('1.0', 'Type or paste a message here…')
            self.txt.config(fg=MUTED)

    def _load(self, msg):
        self.txt.config(fg=TEXT_C)
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', msg)

    def _clear(self):
        self.txt.delete('1.0', 'end')
        self._ph_restore()
        self.vlbl.config(text='Result will appear here.', fg=MUTED, bg=BG)
        self.vf.config(bg=BG)
        self.dblbl.config(text='')
        for r in self.tree.get_children(): self.tree.delete(r)

    # ── Classification ─────────────────────────────────────────────────────
    def _classify(self):
        if not self.ready:
            messagebox.showinfo('Please wait', 'Models are still loading…'); return
        msg = self.txt.get('1.0', 'end-1c').strip()
        if not msg or msg == 'Type or paste a message here…':
            messagebox.showwarning('Empty', 'Please enter a message.'); return
        self.btn.config(state='disabled')
        self.prog.pack(pady=4); self.prog.start(10)
        threading.Thread(target=self._worker, args=(msg,), daemon=True).start()

    def _worker(self, msg):
        error_msg = None
        results   = None
        try:
            results = classify_message(msg, self.tfidf, self.le, self.scaler, self.models)
        except Exception as ex:
            error_msg = str(ex)
        self.prog.stop(); self.prog.pack_forget()
        self.btn.config(state='normal')
        if error_msg:
            captured = error_msg
            self.after(0, lambda: messagebox.showerror('Error', captured))
        else:
            self.after(0, lambda: self._show(results, msg))

    def _show(self, results, msg):
        for r in self.tree.get_children(): self.tree.delete(r)

        from collections import Counter
        valid  = {k: v for k, v in results.items() if v['label'] != 'ERROR'}
        if not valid: return
        votes  = [v['label'] for v in valid.values()]
        winner = Counter(votes).most_common(1)[0][0]
        agree  = votes.count(winner)

        color = LABEL_COLORS.get(winner, ACCENT)
        icon  = LABEL_ICONS.get(winner, '❓')
        self.vf.config(bg=color)
        self.vlbl.config(
            text=f'{icon}  VERDICT: {winner}  ({agree}/{len(votes)} models agree)',
            fg='white', bg=color,
            font=('Helvetica', 16, 'bold'))

        for name, res in results.items():
            lbl  = res['label']
            p    = res['probs']
            icon2 = LABEL_ICONS.get(lbl, '')
            self.tree.insert('', 'end', values=(
                name,
                f'{icon2} {lbl}',
                f"{res['confidence']:.1f}%",
                f"{p.get('ham',   0):.1f}%",
                f"{p.get('spam',  0):.1f}%",
                f"{p.get('smishing', 0):.1f}%",
            ), tags=(lbl,))

        # Debug: show which hand-crafted features fired
        feats = extract_features(msg)
        names = [
            'url','email','phone','brand','smish_kw','smish_only','spam_kw','ham_kw',
            'brand+url','smish+url','spam-url','spam-brand',
            'pure_spam','pure_smish','spam-personal','ham-spam',
            'urgency','caps%','excl','words','shorturl','rawphone',
            'link+urg','£','$','casual'
        ]
        fired = [f'{n}={v:.2f}' if isinstance(v, float) and v != int(v)
                 else f'{n}={int(v)}' if int(v) else None
                 for n, v in zip(names, feats)]
        fired = [f for f in fired if f]
        self.dblbl.config(text='Active features: ' + '  |  '.join(fired) if fired
                          else 'No features fired.')


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = App()
    app.mainloop()
