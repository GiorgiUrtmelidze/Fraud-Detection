# IEEE-CIS Fraud Detection


[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) წარმოადგენს ბინარული კლასიფიკაციის (binary classification) ამოცანას, სადაც UnitedHealth-ისა და IEEE-ის მონაცემებზე დაყრდნობით უნდა განვსაზღვროთ, არის თუ არა ტრანზაქცია თაღლითური (`isFraud=1`).

მონაცემები ორ ძირითად ცხრილადაა დაყოფილი:
* **`transaction`** (ფინანსური ინფორმაცია): მოიცავს ისეთ ცვლადებს, როგორებიცაა `TransactionAmt`, `card1..card6`, `addr1/addr2`, `dist1/dist2`, `D1..D15`, `V1..V339`, `C1..C14` და სხვა მახასიათებლები.
* **`identity`** (ინფორმაცია მოწყობილობისა და ბრაუზერის შესახებ): მოიცავს `id_01..id_38`, `DeviceType` და `DeviceInfo` სვეტებს.

ცხრილები ერთიანდება `TransactionID` სვეტის მეშვეობით. ვინაიდან `identity` ცხრილი ტრანზაქციების მხოლოდ მცირე ნაწილს მოიცავს, მონაცემთა გასაერთიანებლად გამოყენებულია `LEFT JOIN`.

შეფასების მეტრიკაა **ROC AUC**. სამიზნე ცვლადი ხასიათდება მკვეთრი დისბალანსით (თაღლითური ტრანზაქციების წილი ~3.5%-ია).

## პრობლემის გადაჭრა

1.  **(Cleaning)** — წაიშალა ის სვეტები, რომლებშიც გამოტოვებული მნიშვნელობების (`NaN`) წილი 90%-ს აღემატებოდა. გამონაკლისის სახით დავტოვე `dist1/dist2` და `D1..D5` სვეტები, რადგან, მიუხედავად მონაცემთა სიმცირისა, ისინი მნიშვნელოვან ინფორმაციას შეიცავს.
2.  **(Feature Engineering)** — მოხდა ელ-ფოსტის დომენების დაჯგუფება (მაგ: gmail/google → `google`), `TransactionAmt`-ისთვის გამოითვლება ლოგარითმული მნიშვნელობა და ათწილადი ნაწილი, ხოლო `TransactionDT` სვეტიდან აღდგა დროის ერთეულები (საათი, კვირის დღე).
3.  **(Categorical Encoding)** — გამოყენებულია Frequency Encoding (rare-value collapsing). ამ მეთოდმა XGB/LGBM მოდელებისთვის ვალიდაციისას (CV) Label Encoding-ზე უკეთესი შედეგი აჩვენა.
4.  **(Feature Selection)** — ჩატარდა კორელაციური ფილტრაცია (|r|>0.95) და Variance Threshold-ის შემოწმება. საბოლოო ეტაპზე მახასიათებლები შეირჩა მოდელზე დაფუძნებული მნიშვნელოვნების (Feature Importance) მიხედვით. XGBoost-ისთვის ოპტიმალური აღმოჩნდა  200 მახასიათებლის გამოყენება.
5.  **(Training)** — გამოყენებულია 5-fold StratifiedKFold კროს-ვალიდაცია. ჰიპერპარამეტრების ოპტიმიზაცია განხორციელდა Optuna-ს მეშვეობით, რის შემდეგაც მოდელი ხელახლა დატრენინგდა (refit) სრულ სატრენინგო სიმრავლეზე.
6.  **პროცესის ავტომატიზაცია** — ყველა ნაბიჯი მოქცეულია `sklearn`-ის პაიპლაინში, რაც საშუალებას გვაძლევს raw მონაცემები პირდაპირ მივაწოდოთ მოდელს წინასწარი დამუშავების გარეშე.

## პროექტის სტრუქტურა

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── preprocessing.py          # დამხმარე სკრიპტები 
├── scripts/
│   └── build_notebooks.py        # Notebook-ების გენერატორი (utility)
└── notebooks/
    ├── model_experiment_LogisticRegression.ipynb
    ├── model_experiment_RandomForest.ipynb
    ├── model_experiment_XGBoost.ipynb
    ├── model_experiment_LightGBM.ipynb
    ├── model_experiment_CatBoost.ipynb
    └── model_inference.ipynb
