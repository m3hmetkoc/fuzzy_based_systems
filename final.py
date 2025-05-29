import numpy as np

class FuzzySet:
    def __init__(self, name, universe_range):
        self.name = name
        self.universe = np.arange(universe_range[0], universe_range[1], universe_range[2])
        self.membership_functions = {}

    def add_mf(self, name, mf_type, params):
        """Add a membership function to the fuzzy set"""
        self.membership_functions[name] = (mf_type, params)

    def calculate_membership(self, name, x):
        """Calculate membership value for a given input"""
        mf_type, params = self.membership_functions[name]

        if mf_type == 'trimf':
            return self._trimf(x, params)
        elif mf_type == 'trapmf':
            return self._trapmf(x, params)
        return 0

    def _trimf(self, x, params):
        """Triangular membership function"""
        a, b, c = params
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    def _trapmf(self, x, params):
        """Trapezoidal membership function"""
        a, b, c, d = params
        if x <= a or x >= d:
            return 0
        elif b <= x <= c:
            return 1
        elif a < x < b:
            return (x - a) / (b - a)
        else:
            return (d - x) / (d - c)


class FuzzyInferenceSystem:
    def __init__(self):
        self.input_sets = {}
        self.output_sets = {}
        self.rules = []

    def add_input(self, name, universe_range):
        """Add an input variable"""
        self.input_sets[name] = FuzzySet(name, universe_range)
        return self.input_sets[name]

    def add_output(self, name, universe_range):
        """Add an output variable"""
        self.output_sets[name] = FuzzySet(name, universe_range)
        return self.output_sets[name]

    def add_rule(self, antecedents, consequent):
        """Add a fuzzy rule"""
        self.rules.append((antecedents, consequent))

    def evaluate(self, inputs):
        """Evaluate the FIS for given inputs"""
        # Calculate rule strengths
        rule_strengths = []
        for antecedents, consequent in self.rules:
            strength = 1.0
            for var_name, mf_name in antecedents:
                x = inputs[var_name]
                membership = self.input_sets[var_name].calculate_membership(mf_name, x)
                strength = min(strength, membership)
            rule_strengths.append((strength, consequent))

        # Defuzzification using center of gravity method
        if not rule_strengths:
            return 0

        output_set = list(self.output_sets.values())[0]
        numerator = 0
        denominator = 0

        for x in output_set.universe:
            max_membership = 0
            for strength, (output_name, mf_name) in rule_strengths:
                membership = min(strength, output_set.calculate_membership(mf_name, x))
                max_membership = max(max_membership, membership)

            numerator += x * max_membership
            denominator += max_membership

        return numerator / denominator if denominator != 0 else 0


def create_house_fis():
    fis = FuzzyInferenceSystem()

    # Market Values
    market = fis.add_input('market_value', (0, 1000001, 1000))
    market.add_mf('low', 'trapmf', [0, 0, 80000, 100000])
    market.add_mf('medium', 'trapmf', [50000, 100000, 200000, 250000])
    market.add_mf('high', 'trapmf', [200000, 300000, 700000, 850000])
    market.add_mf('very_high', 'trapmf', [650000, 850000, 1000000, 1000000])


    # Location
    location = fis.add_input('location', (0, 11, 0.5))
    location.add_mf('bad', 'trapmf', [0, 0, 2, 4])
    location.add_mf('fair', 'trapmf', [2.5, 5, 6, 8.5])
    location.add_mf('excellent', 'trapmf', [6, 8.5, 10, 10])

    # House Evaluation Output
    house_eval = fis.add_output('house_eval', (0, 11, 1))
    house_eval.add_mf('very_low', 'trimf', [0, 0, 3])
    house_eval.add_mf('low', 'trimf', [0, 3, 6])
    house_eval.add_mf('medium', 'trimf', [2, 5, 8])
    house_eval.add_mf('high', 'trimf', [4, 7, 10])
    house_eval.add_mf('very_high', 'trimf', [7, 10, 10])

    # Add house rules
    fis.add_rule([('market_value', 'low')], ('house_eval', 'low'))
    fis.add_rule([('location', 'bad')], ('house_eval', 'low'))
    fis.add_rule([('location', 'bad'), ('market_value', 'low')], ('house_eval', 'very_low'))
    fis.add_rule([('location', 'bad'), ('market_value', 'medium')], ('house_eval', 'low'))
    fis.add_rule([('location', 'bad'), ('market_value', 'high')], ('house_eval', 'medium'))
    fis.add_rule([('location', 'bad'), ('market_value', 'very_high')], ('house_eval', 'high'))
    fis.add_rule([('location', 'fair'), ('market_value', 'low')], ('house_eval', 'low'))
    fis.add_rule([('location', 'fair'), ('market_value', 'medium')], ('house_eval', 'medium'))
    fis.add_rule([('location', 'fair'), ('market_value', 'high')], ('house_eval', 'high'))
    fis.add_rule([('location', 'fair'), ('market_value', 'very_high')], ('house_eval', 'very_high'))
    fis.add_rule([('location', 'excellent'), ('market_value', 'low')], ('house_eval', 'medium'))
    fis.add_rule([('location', 'excellent'), ('market_value', 'medium')], ('house_eval', 'high'))
    fis.add_rule([('location', 'excellent'), ('market_value', 'high')], ('house_eval', 'very_high'))
    fis.add_rule([('location', 'excellent'), ('market_value', 'very_high')], ('house_eval', 'very_high'))

    return fis


def create_application_fis():
    fis = FuzzyInferenceSystem()

    # Assets
    assets = fis.add_input('assets', (0, 1000001, 1000))
    assets.add_mf('low', 'trimf', [0, 0, 150000])
    assets.add_mf('medium', 'trapmf', [50000, 250000, 450000, 650000])
    assets.add_mf('high', 'trapmf', [500000, 700000, 1000000, 1000000])

    # Salary
    salary = fis.add_input('salary', (0, 100001, 100))
    salary.add_mf('low', 'trapmf', [0, 0, 10000, 25000])
    salary.add_mf('medium', 'trimf', [15000, 35000, 55000])
    salary.add_mf('high', 'trimf', [40000, 60000, 80000])
    salary.add_mf('very_high', 'trapmf', [60000, 80000, 100000, 100000])

    # Application Evaluation Output
    eval_app = fis.add_output('eval_app', (0, 11, 1))
    eval_app.add_mf('low', 'trapmf', [0, 0, 2, 4])
    eval_app.add_mf('medium', 'trimf', [2, 5, 8])
    eval_app.add_mf('high', 'trapmf', [6, 8, 10, 10])

    # Add application rules
    fis.add_rule([('assets', 'low'), ('salary', 'low')], ('eval_app', 'low'))
    fis.add_rule([('assets', 'low'), ('salary', 'medium')], ('eval_app', 'low'))
    fis.add_rule([('assets', 'low'), ('salary', 'high')], ('eval_app', 'medium'))
    fis.add_rule([('assets', 'low'), ('salary', 'very_high')], ('eval_app', 'high'))
    fis.add_rule([('assets', 'medium'), ('salary', 'low')], ('eval_app', 'low'))
    fis.add_rule([('assets', 'medium'), ('salary', 'medium')], ('eval_app', 'medium'))
    fis.add_rule([('assets', 'medium'), ('salary', 'high')], ('eval_app', 'high'))
    fis.add_rule([('assets', 'medium'), ('salary', 'very_high')], ('eval_app', 'high'))
    fis.add_rule([('assets', 'high'), ('salary', 'low')], ('eval_app', 'medium'))
    fis.add_rule([('assets', 'high'), ('salary', 'medium')], ('eval_app', 'medium'))
    fis.add_rule([('assets', 'high'), ('salary', 'high')], ('eval_app', 'high'))
    fis.add_rule([('assets', 'high'), ('salary', 'very_high')], ('eval_app', 'high'))

    return fis


def create_loan_fis():
    fis = FuzzyInferenceSystem()

    # Inputs from previous evaluations
    house_eval = fis.add_input('house_eval', (0, 11, 1))
    house_eval.add_mf('very_low', 'trimf', [0, 0, 3])
    house_eval.add_mf('low', 'trimf', [0, 3, 6])
    house_eval.add_mf('medium', 'trimf', [2, 5, 8])
    house_eval.add_mf('high', 'trimf', [4, 7, 10])
    house_eval.add_mf('very_high', 'trimf', [7, 10, 10])


    eval_app = fis.add_input('eval_app', (0, 11, 1))
    eval_app.add_mf('low', 'trapmf', [0, 0, 2, 4])
    eval_app.add_mf('medium', 'trimf', [2, 5, 8])
    eval_app.add_mf('high', 'trapmf', [6, 8, 10, 10])

    # Interest Rate
    interest = fis.add_input('interest', (0, 10, 0.5))
    interest.add_mf('low', 'trapmf', [0, 0, 2, 5])
    interest.add_mf('medium', 'trapmf', [2, 4, 6, 8])
    interest.add_mf('high', 'trapmf', [6, 8.5, 10, 10])

    # Salary
    salary = fis.add_input('salary', (0, 100001, 100))
    salary.add_mf('low', 'trapmf', [0, 0, 10000, 25000])
    salary.add_mf('medium', 'trimf', [15000, 35000, 55000])
    salary.add_mf('high', 'trimf', [40000, 60000, 80000])
    salary.add_mf('very_high', 'trapmf', [60000, 80000, 100000, 100000])

    # Loan Amount Output
    loan = fis.add_output('loan', (0, 500001, 1000))
    loan.add_mf('very_low', 'trimf', [0, 0, 125000])
    loan.add_mf('low', 'trimf', [0, 125000, 250000])
    loan.add_mf('medium', 'trimf', [125000, 250000, 375000])
    loan.add_mf('high', 'trimf', [250000, 375000, 500000])
    loan.add_mf('very_high', 'trimf', [375000, 500000, 500000])

    # Add loan rules
    fis.add_rule([('salary', 'low'), ('interest', 'medium')], ('loan', 'very_low'))
    fis.add_rule([('salary', 'low'), ('interest', 'high')], ('loan', 'very_low'))
    fis.add_rule([('salary', 'medium'), ('interest', 'high')], ('loan', 'low'))
    fis.add_rule([('eval_app', 'low')], ('loan', 'very_low'))
    fis.add_rule([('house_eval', 'very_low')], ('loan', 'very_low'))
    fis.add_rule([('eval_app', 'medium'), ('house_eval', 'very_low')], ('loan', 'low'))
    fis.add_rule([('eval_app', 'medium'), ('house_eval', 'low')], ('loan', 'low'))
    fis.add_rule([('eval_app', 'medium'), ('house_eval', 'medium')], ('loan', 'medium'))
    fis.add_rule([('eval_app', 'medium'), ('house_eval', 'high')], ('loan', 'high'))
    fis.add_rule([('eval_app', 'medium'), ('house_eval', 'very_high')], ('loan', 'high'))
    fis.add_rule([('eval_app', 'high'), ('house_eval', 'very_low')], ('loan', 'low'))
    fis.add_rule([('eval_app', 'high'), ('house_eval', 'low')], ('loan', 'medium'))
    fis.add_rule([('eval_app', 'high'), ('house_eval', 'medium')], ('loan', 'high'))
    fis.add_rule([('eval_app', 'high'), ('house_eval', 'high')], ('loan', 'high'))
    fis.add_rule([('eval_app', 'high'), ('house_eval', 'very_high')], ('loan', 'very_high'))

    return fis

def evaluate_loan(market_value, location_value, assets_value, salary_value, interest_value):
    # Evaluate house
    house_fis = create_house_fis()
    house_eval = house_fis.evaluate({
        'market_value': market_value,
        'location': location_value
    })

    # Evaluate application
    app_fis = create_application_fis()
    app_eval = app_fis.evaluate({
        'assets': assets_value,
        'salary': salary_value
    })

    # Evaluate loan
    loan_fis = create_loan_fis()
    loan_amount = loan_fis.evaluate({
        'house_eval': house_eval,
        'eval_app': app_eval,
        'interest': interest_value,
        'salary': salary_value
    })

    return {
        'house_evaluation': house_eval,
        'application_evaluation': app_eval,
        'loan_amount': loan_amount
    }


# Test the system
if __name__ == "__main__":
    test_cases = [
        # Scenario 1: very high market value, excellent location, high asset, very high salary, high interest rate
        (900000, 9.5, 800000, 85000, 9),
        # Scenario 2: medium market value, fair location, medium asset, medium salary, medium interest rate
        (180000, 6.0, 250000, 45000, 5),
        # Scenario 3: low market value, bad location, low asset, low salary, low interest rate
        (75000, 2.0, 30000, 20000, 8.5),
        #Scenario 4: medium market value, excellent location, medium asset, low salary, medium interest rate
        (120000, 8.0, 300000, 5000, 6.5),
        #Scenario 5: high market value, fair location, medium asset, high salary, medium interest rate
        (400000, 5.0, 400000, 60000, 4)
    ]
    i = 0
    for market_value, location, assets, salary, interest in test_cases:
        i += 1
        result = evaluate_loan(market_value, location, assets, salary, interest)
        print(f"\nTest Case:", i)
        print(f"Market Value: ${market_value:,}")
        print(f"Location: {location}/10")
        print(f"Assets: ${assets:,}")
        print(f"Salary: ${salary:,}")
        print(f"Interest Rate: {interest}%")
        print("\nResults:")
        print(f"House Evaluation: {result['house_evaluation']:.2f}/10")
        print(f"Application Evaluation: {result['application_evaluation']:.2f}/10")
        print(f"Recommended Loan Amount: ${result['loan_amount']:,.2f}")
        print("-" * 50)