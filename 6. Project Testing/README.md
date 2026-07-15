## 6. Project Testing

### 6.1. Testing Methodology
The verification system utilizes **Unit Testing** via Python's standard `unittest` framework. The validation focus centers on:
1. Routing status check.
2. Boundary and constraints verification.
3. Predict logic validation (confirming expected model responses for typical approved and declined applicant profiles).

### 6.2. Automated Test Configuration (`test_app.py`)
```python
# Code sample from test_app.py
class CreditCardAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction_approved_case(self):
        payload = {'age': '35', 'income': '120000', 'debt': '0', 'credit_score': '820'}
        response = self.app.post('/get_prediction', data=payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Approved', response.data)
```

### 6.3. Test Results Table

| Test Case ID | Test Component | Input Payload | Expected Output | Actual Output | Status |
|---|---|---|---|---|---|
| TC-01 | Home Route Loading | GET `/` | Status 200, contains "Credit Shield AI" | Status 200, text verified | **PASS** |
| TC-02 | Form Page Loading | GET `/predict` | Status 200, contains "Applicant Profile" | Status 200, text verified | **PASS** |
| TC-03 | Approval Prediction | Age=35, Income=120k, Debt=0, FICO=820 | Page renders "Approved" | Page renders "Approved" | **PASS** |
| TC-04 | Rejection Prediction | Age=22, Income=15k, Debt=20k, FICO=350 | Page renders "Declined" | Page renders "Declined" | **PASS** |
| TC-05 | Boundary Validation | FICO=900 (Invalid) | Renders: "Credit Score must be 300-850" | Renders validation warning | **PASS** |

---