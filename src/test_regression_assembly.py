import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest

def test_linear_regression_coefs():
    # y = 5 + 2*x1 - 3*x2 exactly
    rng = np.random.default_rng(0)
    x1 = rng.random(100)
    x2 = rng.random(100)
    y  = 5 + 2*x1 - 3*x2
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    X  = sm.add_constant(df[['x1', 'x2']])
    ols = sm.OLS(df['y'], X).fit()
    params = ols.params
    assert params['const'] == pytest.approx(5.0, abs=1e-6)
    assert params['x1']    == pytest.approx(2.0, abs=1e-6)
    assert params['x2']    == pytest.approx(-3.0, abs=1e-6)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])