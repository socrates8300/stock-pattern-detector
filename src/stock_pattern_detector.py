import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kdbai import Client

# API Keys
FMP_API_KEY = "58IG1ii3BEe63GZkW2Vn6SBzNFLQQaiP"
KDBAI_API_KEY = "5f6243346f-j/jNLGF3i1ASBIVuI6kdam/MsBc6+M5nV+Pp0Y5t6ANEQTbHPX1LFYKBbdCID0hOYa8ReYOjbmT9waJd"

class StockPatternDetector:
    def __init__(self):
        self.kdb_client = Client(api_key=KDBAI_API_KEY)
        
    def get_stock_data(self, symbol, from_date, to_date):
        """Fetch stock data from Financial Modeling Prep API"""
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {
            "apikey": FMP_API_KEY,
            "from": from_date,
            "to": to_date
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "historical" in data:
                df = pd.DataFrame(data["historical"])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                return df
        return None

    def prepare_data_for_tss(self, df, window_size=180):
        """Prepare data for Temporal Similarity Search"""
        # Convert price data to numpy array for TSS
        prices = df['close'].values
        
        # Create sliding windows
        windows = []
        for i in range(len(prices) - window_size + 1):
            window = prices[i:i + window_size]
            # Normalize the window
            window = (window - np.mean(window)) / np.std(window)
            windows.append(window)
            
        return np.array(windows)

    def detect_anomalies(self, historical_data, live_data, threshold=0.8):
        """
        Detect anomalies using TSS by comparing live data patterns
        against historical patterns
        """
        # Use KDB.AI's TSS to find similar patterns
        results = self.kdb_client.search(
            collection="stock_patterns",
            query_vector=live_data,
            k=5  # Number of nearest neighbors to return
        )
        
        # If similarity scores are below threshold, consider it an anomaly
        similarities = [result.score for result in results]
        return max(similarities) < threshold

    def identify_patterns(self, data, pattern_template):
        """
        Search for specific patterns in the data using TSS
        """
        # Use KDB.AI's TSS to find matches to the pattern template
        results = self.kdb_client.search(
            collection="stock_patterns",
            query_vector=pattern_template,
            k=5
        )
        return results

def main():
    detector = StockPatternDetector()
    
    # Example usage
    symbol = "TSLA"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    # Get historical data
    df = detector.get_stock_data(
        symbol,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    if df is not None:
        # Prepare data for TSS
        windows = detector.prepare_data_for_tss(df)
        
        # Example of pattern detection
        # Here we could implement real-time monitoring of new data
        # and pattern detection using the prepared windows
        print(f"Prepared {len(windows)} windows for pattern detection")
        
        # TODO: Implement real-time monitoring and pattern detection
        # This would involve:
        # 1. Continuously fetching new data
        # 2. Creating windows from new data
        # 3. Using TSS to detect patterns and anomalies
        # 4. Generating alerts when patterns or anomalies are found

if __name__ == "__main__":
    main()