dd = reversion.equity / reversion.equity.cummax() - 1.0
print(dd.min(), dd.notna().sum())   # expect ~ -0.14 and a full count
