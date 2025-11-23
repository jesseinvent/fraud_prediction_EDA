import math
import os
import pandas as pd
import numpy as np
from typing import TypedDict, Optional
from collections import Counter
from datetime import datetime
from pathlib import Path
from enum import Enum


class TransactionType(Enum):
    LIGHTNING_INVOICE = "lightning_invoice"
    LIGHTNING_LNURL = 'lightning_lnurl'
    VAS = "vas"
    ONCHAIN = "onchain"

class SourceDataType(TypedDict):
    email: str
    userAccountName: str
    accountCreationDate: str # date
    isKycVerified: bool
    lastRequestIpAddress: str
    currentRequestIpAddress: str
    currentTransactionDate: str # date
    transactionType: TransactionType # ['lightning_invoice', 'lightning_lnurl', 'vas','onchain']
    currentTransactionAmount: int
    lastTransactionDate: str # date
    lastTransactionAmount: int
    numberOfTransactionsInLast24h: int
    last5TransactionsAmount: list[float]
    lastTransactionDestination: str
    currentTransactionDestination: str
    walletBalance: float
    isFraud: bool
    
    '''
        - User signals (emailEntropy, userAccountNameEntropy, dummy checks, disposable email detection, KYC verification, account age).
        
        •	Behavioral patterns (timeSinceLastTransactionLog, numberOfTransactionsInLast24h,              lastTransactionAmountIsTheSame, transaction time features).
        
        •	Transaction properties (transactionAmountLog, transactionAmountRelativeToMedianLog, walletBalanceIsSufficient).
        •	Temporal cyclic encodings (sin/cos for day/hour).
        •	Network / device context (ipAddressIsDifferentFromLastRequestIp).
        •	Label (isFraud).
    '''
class TransactionFeatures(TypedDict):
    emailEntropy: float
    emailIsDisposable: int
    userAccountNameEntropy: float
    isDummyName: int
    isDummyEmail: int
    accountAgeLogBeforeTransaction: float
    accountIsNew: int
    kycVerified: int
    ipAddressIsDifferentFromLastRequestIp: int
    transactionTimeIsWeekend: int
    transactionTimeIsSleepingHours: int
    
    transactionType_lightningLnurl: int
    # transactionType_lightningInvoice: int
    transactionType_vas: int
    transactionType_onchain: int
    
    dayOfTransactionSin: float
    dayOfTransactionCos: float
    hourOfTransactionSin: float
    hourOfTransactionCos: float
    timeSinceLastTransactionLog: float
    lastTransactionAmountIsTheSame: int
    transactionAmountRelativeToMedianLog: float
    numberOfTransactionsInLast24h: int
    transactionAmountLog: float
    walletBalanceIsSufficient: int
    isFraud: Optional[int] # remove during prediction
    
transactionFeaturesTypeSchema = {
    "emailEntropy": "float32",
    "emailIsDisposable": "int8",
    "userAccountNameEntropy": "float32",
    "isDummyName": "int8",
    "isDummyEmail": "int8",
    "accountAgeLogBeforeTransaction": "float32",
    "accountIsNew": "int8",
    "kycVerified": "int8",
    "ipAddressIsDifferentFromLastRequestIp": "int8",
    "transactionTimeIsWeekend": "int8",
    "transactionTimeIsSleepingHours": "int8",
    
    "transactionType_lightningLnurl": "int8",
    "transactionType_lightningInvoice": "int8",
    "transactionType_vas": "int8",
    "transactionType_onchain": "int8",
    
    "dayOfTransactionSin": "float32",
    "dayOfTransactionCos": "float32",
    "hourOfTransactionSin": "float32",
    "hourOfTransactionCos": "float32",
    "timeSinceLastTransactionLog": "float32",
    "lastTransactionAmountIsTheSame": "int8",
    "transactionAmountRelativeToMedianLog": "float32",
    "numberOfTransactionsInLast24h": "int8",
    "transactionAmountLog": "float32",
    "walletBalanceIsSufficient": "int8",
    "isFraud": "int8" # remove during prediction
}    

json_path = os.path.join(BASE_DIR, "dataset", "disposable-email-domains.json")

disposableEmailDomains = pd.read_json(json_path)
    
def isDisposableEmail(email: str) -> int:
            
    emailDomain = email.split('@')[1].lower()
    
    if emailDomain in disposableEmailDomains['disposable_email_domains'].values: return 1
    
    return 0

def calculateStringEntropy(string: str) -> float:
    '''
        Entropy comes from information theory. For a string (like a username), it roughly measures how “random” the string is:
        •	A username like "johnsmith" → low entropy (common, predictable)
        •	A username like "X9fK2!bQ" → high entropy (random, complex)
        Entropy is often calculated as:

        H = - \sum p_i \log_2 p_i
        
        •	Low values → normal users
	    •	High values → suspicious, potentially fraudulent accounts
    '''
    '''
        Counts how many times each character appears in the string.
        E.G
            s = "hello"
            p = Counter(s)
            print(p)  
            # Output: Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
    '''
    characterCount = Counter(string)
    totalCharacters = len(string)
    
    entropy = 0.0
    
    for count in characterCount.values():
        # Calculate probability of each character
        probability = count / totalCharacters
    
        entropy += probability * math.log2(probability)
    
    return -entropy
    
def isDummyEmail(email: str) -> int:
    '''
        A disposable email with a high entropy
    '''
    emailEntropy = calculateStringEntropy(email.split('@')[0])
    
    if isDisposableEmail(email): return 1 
        
    if emailEntropy > 3.1: return 1
    
    if isDisposableEmail(email) and emailEntropy > 2.5: return 1
    
    return 0  

# TODO: Check KYC status
def isDummyName(name: str) -> int: 
    '''
        Names with high entropy
    '''
    
    return 1 if calculateStringEntropy(name) > 3.1 else 0

def calculateAccountAgeLogBeforeTransaction(signupDate: str, transactionDate: str) -> float:
    '''
        Calculates how old an account is before transaction
    '''
    created = pd.to_datetime(signupDate, utc=True)
    transactionTime = pd.to_datetime(transactionDate, utc=True)
    
    ageDays = (transactionTime - created).total_seconds() / (60 * 60 * 24) # in Days
    
    if ageDays < 0: return 0
    
    return math.log1p(ageDays)

def isAccountNew(signupDate: str) -> bool:
    '''
        Check if account is less than a day old
    '''
    created = pd.to_datetime(signupDate, utc=True)
    
    now = pd.to_datetime(datetime.now(), utc=True)
    
    accountAgeInDays =  (now - created).total_seconds() / (60 * 60 * 24)
    
    if accountAgeInDays < 1: return 1
    
    return 0

def isTransactionTimeWeekend(transactionDate: str) -> int:
    '''
        Checks if transaction date is during the weekend
    '''
    if pd.to_datetime(transactionDate, utc=True).weekday() >= 5: return 1 
    
    return 0
    
def isTransactionTimeSleepingHours(transactionDate: str) -> int:
    '''
        Checks if transaction time is sleeping hours (22:00 - 5:00)
    '''        
    hour = pd.to_datetime(transactionDate, utc=True).hour
        
    if hour >= 22 or hour <= 5: return 1 
    
    return 0

def encodeTransactionType(transactionType: str) -> dict:
    return {
        "transactionType_lightningLnurl": 1 if transactionType == "lightning_lnurl" else 0,
        # "transactionType_lightningInvoice": 1 if transactionType == "lightning_lnurl" else 0,
        "transactionType_vas": 1 if transactionType == "vas" else 0,
        "transactionType_onchain": 1 if transactionType == "onchain" else 0,
    }

def dayOfTransactionSinCos(transactionDate: str):
    '''
        dayOfTransactionSin and dayOfTransactionCos are used to represent the day of the week as a cyclical feature. 
        This is important because days repeat every 7 days, and encoding them as 0–6 does not capture the cyclic nature.
    '''
    weekDay = pd.to_datetime(transactionDate, utc=True).weekday()
    
    sinVal = math.sin(2 * math.pi * weekDay / 7)
    cosVal = math.cos(2 * math.pi * weekDay / 7)
    
    return sinVal, cosVal

def hourOfTransactionSinCos(transactionDate: str):
    '''
        Calculates sine and cosine of transaction hour for cyclical encoding.
    '''
    hour = pd.to_datetime(transactionDate, utc=True).hour
    
    sinVal = math.sin(2 * math.pi * hour / 7)
    cosVal = math.cos(2 * math.pi * hour / 7)
    
    return sinVal, cosVal

def calculateTimeSinceLastTransactionLog(currentTransactionDate: str, lastTransactionDate: str) -> float:
    '''
        Calculate time differences from current and last transaction
    '''
    if not lastTransactionDate:  # first transaction, no history
        return 0.0   # or np.nan depending on how you want to model it
    
    current = pd.to_datetime(currentTransactionDate, utc=True)
    last = pd.to_datetime(lastTransactionDate, utc=True)
    
    if pd.isna(last):
        return 0.0
  
    diff_hours = (current - last).total_seconds() / 3600  # convert seconds → hours
    
    # ✅ Prevent negatives (e.g. bad/misordered data)
        
    if diff_hours is None or diff_hours < 0:
        return 0.0  # or np.nan to indicate invalid ordering
    
    return math.log1p(diff_hours)  # log(1 + hours)

def calculateTransactionAmountRelativeToMedianLog(currentTransactionAmount: float, pastTransactionAmounts: list[float]) -> float:
    '''
        transactionAmountRelativeToMedianLog usually measures how different the current transaction amount is 
        compared to the typical (median) transaction amount for that user.
        Detects out-of-pattern transaction sizes
    '''
    
    pastTransactionAmounts = [x if x is not None else 0 for x in pastTransactionAmounts] #convert null to 0
    
    if len(pastTransactionAmounts) == 0:
        return 0.0
    
    medianAmount = np.median(pastTransactionAmounts)
    
    if medianAmount is None or medianAmount <= 0:
        return 0.0
    
    return math.log(currentTransactionAmount / medianAmount)

def enforceFeaturesDataTypes(features):
    for col, dtype in transactionFeaturesTypeSchema.items():
        if col in features.columns:
            features[col] = features[col].astype(dtype)
            
    return features        
  
def extractFeatureSetFromSourceData(sourceData: SourceDataType, includeLabel: bool = False) -> TransactionFeatures:

    features: TransactionFeatures = {
        'emailEntropy': 0.1,
        'emailIsDisposable': isDisposableEmail(sourceData['email']),
        'userAccountNameEntropy': calculateStringEntropy(sourceData['userAccountName'].strip()),
        'isDummyName': isDummyName(sourceData['userAccountName']),
        'isDummyEmail': 0 if bool(sourceData['isKycVerified']) is True else int(isDummyEmail(sourceData['email'])),
        'accountAgeLogBeforeTransaction': calculateAccountAgeLogBeforeTransaction(sourceData['accountCreationDate'], sourceData['currentTransactionDate']),
        'accountIsNew': isAccountNew(sourceData['accountCreationDate']),
        'kycVerified': 1 if bool(sourceData["isKycVerified"]) is True else 0,
        'ipAddressIsDifferentFromLastRequestIp': 1 if sourceData['lastRequestIpAddress'] == sourceData['currentRequestIpAddress'] else 0,
        'transactionTimeIsWeekend': isTransactionTimeWeekend(sourceData['currentTransactionDate']),
        'transactionTimeIsSleepingHours': isTransactionTimeSleepingHours(sourceData['currentTransactionDate']),
        **encodeTransactionType(sourceData['transactionType']),
        'dayOfTransactionSin': dayOfTransactionSinCos(sourceData['currentTransactionDate'])[0],
        'dayOfTransactionCos': dayOfTransactionSinCos(sourceData['currentTransactionDate'])[1],
        "hourOfTransactionSin": hourOfTransactionSinCos(sourceData['currentTransactionDate'])[0],
        "hourOfTransactionCos": hourOfTransactionSinCos(sourceData['currentTransactionDate'])[1],
        "timeSinceLastTransactionLog": calculateTimeSinceLastTransactionLog(sourceData['currentTransactionDate'], sourceData['lastTransactionDate']),
        'lastTransactionAmountIsTheSame': 1 if sourceData["currentTransactionAmount"] == sourceData["lastTransactionAmount"] else 0,
        'transactionAmountRelativeToMedianLog': calculateTransactionAmountRelativeToMedianLog(sourceData['currentTransactionAmount'], sourceData['last5TransactionsAmount']),
        'numberOfTransactionsInLast24h': sourceData['numberOfTransactionsInLast24h'],
        'transactionAmountLog': math.log1p(sourceData['currentTransactionAmount']),
        'walletBalanceIsSufficient': 1 if int(sourceData['walletBalance'] > sourceData['currentTransactionAmount']) else 0,
    }   
    
    if includeLabel == True:
        features['isFraud'] = sourceData['isFraud']
        
    return features    