import pandas as pd

SEVERITY_MAP = {
    'Normal': 0, 'Fuzzers': 3, 'Reconnaissance': 4, 'Generic': 5, 'DoS': 6,
    'Analysis': 6, 'Exploits': 7, 'Backdoor': 8, 'Shellcode': 9, 'Worms': 10
}

def map_severity_score(df: pd.DataFrame, attack_column='attack_cat') -> pd.DataFrame:
    df['severity_score'] = df[attack_column].map(SEVERITY_MAP)
    df['severity_score'] = df['severity_score'].fillna(-1)
    return df
