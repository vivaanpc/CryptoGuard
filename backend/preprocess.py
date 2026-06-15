import re
#regex file
def preprocess_text(text: str) -> str:

    #lowercase
    text = str(text).lower()

    #Handle Hyperlinks using httpurl
    text = re.sub(r"http\S+|www\S+|bit\.ly\S*", "httpurl", text)

    #usermentions
    text = re.sub(r"@\w+", "usermention", text)

    #tickers
    text = re.sub(r"\$([a-zA-Z]+)", r"ticker_\1", text)

    #only valid alphanumeric
    text = re.sub(r"[^a-z0-9_\s]", "", text)

    #spacing cleanse
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


if __name__ == "__main__":
    examples = [
        "SCAM TOKEN just launched!! 1000x guaranteed!! bit.ly/scam",
        "@elonmusk Ethereum 2.0 staking rewards are great #crypto #DeFi",
        "URGENT: Your Binance account will be suspended! Verify at https://binance-secure.xyz",
        "Bitcoin testing $45k resistance. RSI oversold on daily. #BTC $BTC",
    ]
    
    print("Results:")
    for ex in examples:
        print(f"BEFORE: {ex}")
        print(f" AFTER: {preprocess_text(ex)}")
