Titanic Survival Prediction Agent
Μαρία Χαριλάου 20238534

Αυτό το project υλοποιεί έναν AI Agent που προβλέπει την πιθανότητα επιβίωσης επιβάτη του Τιτανικού, εξηγεί την πρόβλεψη και παρέχει συμβουλές και ανατροφοδότηση με βάση τα χαρακτηριστικά του επιβάτη.

---

Περιεχόμενα:

- `titanic_ml.py`: Εκπαίδευση μοντέλου με καθαρισμένα δεδομένα και manual encoding
- `ai_agent.py`: Διαλογικός πράκτορας (agent) που κάνει προβλέψεις και εξηγήσεις
- `model.pkl`: Εκπαιδευμένο μοντέλο RandomForest
- `titanic.csv`: Dataset επιβατών του Τιτανικού

---

Απαραίτητες Βιβλιοθήκες:

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn joblib
```

---

Οδηγίες Εκτέλεση:

Εκπαίδευση Μοντέλου:

Τρέξε το αρχείο `titanic_ml.py` για να:

- Κάνεις καθαρισμό δεδομένων και feature engineering
- Κωδικοποιήσεις τις μεταβλητές `Sex` και `Embarked` χειροκίνητα (χωρίς encoders)
- Εκπαιδεύσεις RandomForest μοντέλο
- Αποθηκεύσεις το `model.pkl`

```bash
python titanic_ml.py
```

---

Εκτέλεση AI Agent:

Μετά την εκπαίδευση, τρέξε το `ai_agent.py`:

```bash
python ai_agent.py
```

Ο agent θα:

- Ζητήσει στοιχεία για νέο επιβάτη
- Υπολογίσει την πιθανότητα επιβίωσης με threshold 0.58
- Δείξει σημαντικά χαρακτηριστικά με SHAP
- Δώσει συμβουλές και "What-if" ανάλυση
- Παρουσιάσει ανατροφοδότηση σε φυσική γλώσσα

---

Παράδειγμα Input στον Agent:

```
Pclass: 3
Sex: male
Age: 30
Siblings/Spouses aboard: 0
Parents/Children aboard: 0
Fare: 7.25
Embarked (C/Q/S): S
```

---

Χαρακτηριστικά του Agent:

- Επεξήγηση εισόδων στον χρήστη
- SHAP-based ανάλυση χαρακτηριστικών
- Custom threshold για καλύτερη ακρίβεια
- Δεν απαιτεί εξωτερικά encoders (.pkl)
- Τελείως αυτόνομος

---

Αρχεία που δημιουργούνται

  Αρχείο           Περιγραφή                            
|---------------|---------------------------------------|
| `model.pkl`   | Εκπαιδευμένο RandomForest μοντέλο     |
| `ai_agent.py` | Agent με προβλέψεις, εξήγηση & feedback |
| `titanic_ml.py` | Εκπαίδευση μοντέλου με preprocessing |
| `titanic.csv` | Δεδομένα Titanic από Kaggle / OpenML |

---

Σημειώσεις:

- Δεν απαιτούνται `LabelEncoders`
- Ο Agent είναι 100% CLI (διαλογικός)
- Μπορεί να επεκταθεί εύκολα σε Web App (π.χ. με Streamlit)

---

Προτάσεις επέκτασης:

- Web interface με κουμπιά και dropdown επιλογές
- Ενσωμάτωση LLM (π.χ. ChatGPT) για φυσική εξήγηση
- Πολλαπλοί χρήστες / batch prediction


