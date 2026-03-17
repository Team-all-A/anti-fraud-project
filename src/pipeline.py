from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

def build_pipeline(model=None) -> Pipeline:
    """
    Будує пайплайн. Аліна передає свою модель:
    build_pipline(model=LGBMClassifier(...))
    """
    if model is None:
        model = GradientBoostingClassifier() #temporary plug

    pipline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return pipline