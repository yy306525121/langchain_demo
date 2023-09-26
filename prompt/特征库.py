from feast import FeatureStore

# 根据存储位置更新路径
feast_repo_path = "../feature_repo/"
store = FeatureStore(repo_path=feast_repo_path)