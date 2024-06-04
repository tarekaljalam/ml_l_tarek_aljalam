#- pandas : تسمح لنا بتحليل وتعديل البيانات بصيغة DataFrame.
import pandas as pd
#   - StandardScaler و OneHotEncoder : تُستخدم لتحويل المتغيرات العددية والفئوية على التوالي.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#  - LogisticRegression و RandomForestClassifier : هما نماذج تعلم آلي يمكن استخدامها للتدريب والتوقعات.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#   - make_pipeline : يُستخدم لبناء أنابيب التدريب بسهولة.
from sklearn.pipeline import make_pipeline
#   - train_test_split و GridSearchCV و cross_val_score : تُستخدم لتقسيم البيانات واختيار أفضل النماذج وتقييمها.
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#- نقوم بقراءة ملف CSV باستخدام pd.read_csv() وتخزين البيانات في متغير data.
data = pd.read_csv("train.csv")

#- نقوم بمعالجة القيم المفقودة في عمود 'Age' عن طريق وضع القيمة المتوسطة باستخدام
data['Age'].fillna(data['Age'].median(), inplace=True)

#  - نقوم بمعالجة القيم المفقودة في عمود 'Cabin' عن طريق وضع القيمة 'Unknown'
data['Cabin'].fillna('Unknown', inplace=True)

#   - نقوم بمعالجة القيم المفقودة في عمود 'Embarked' عن طريق وضع القيمة الأكثر تكرارًا باستخدام 
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# نقوم بتحويل عمود 'Fare' إلى قيمة رقمية
data['Fare'] = pd.to_numeric(data['Fare'], errors='coerce')

# نقوم بإزالة الصفوف التي تحتوي على NaN في عمود 'Fare'
data.dropna(subset=['Fare'], inplace=True)

#  نقوم بإزالة الصفوف المكررة باستخدام 
data.drop_duplicates(inplace=True)

# Feature Scaling: Normalization

# نقوم بإنشاء كائن StandardScaler
StandardScaler = StandardScaler()

# نقوم بتحويل المتغيرات العددية 'Age' و 'Fare' 
NUMERIC = ['Age', 'Fare']
data[NUMERIC] = StandardScaler.fit_transform(data[NUMERIC])

# Categorical Data: One-Hot Encoding البيانات الفئوية: ترميز واحد ساخن
# Apply one-hot encoding to categorical featuresتطبيق ترميز واحد ساخن على الميزات الفئوية
CATEGORI_Feature = ["Sex", "Embarked", "Cabin"]
#نقوم بتطبيق التحويل الفئوي باستخدام
ENCODE_Feature = pd.get_dummies(data[CATEGORI_Feature])
# نقوم بدمج الأعمدة المشفرة مع البيانات الأصلية باستخدام
data = pd.concat([data, ENCODE_Feature], axis=1)

#  نقوم بإزالة الأعمدة الفئوية الأصلية باستخدام
data.drop(CATEGORI_Feature, axis=1, inplace=True)


#تحديد المتغيراتFeature s والهدفTarget 
# نقوم بتعيين المتغيرات في متغير X
X = data.drop(['Survived', 'Name', 'Ticket'], axis=1)  
#نقوم بتعيين الهدف في متغير y
y = data['Survived']

#نقوم بتقسيم البيانات إلى مجموعة التدريب ومجموعة الاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)

# يتم تحديد نموذجين مختلفين للتعلم الآلي تم استخدام LogisticRegression و RandomForestClassifier
Model_One = LogisticRegression()
Model_Two = RandomForestClassifier()

# يتم تعريف الهايبربارامترات التي ستُجرى عليها التجارب لكل نموذج
PARAM_Model_One = {'C': [0.1, 1, 10]}
PARAM_Model_Two = {'n_estimators': [100, 200, 300]}





# يتم إجراء بحث شبكي (Grid Search) للنموذج الأول (Model_One) باستخدام GridSearchCV وتمريرها Model_One و PARAM_Model_One و cv=5 لتحديد عدد الطيات (folds) في التقسيم المتقاطع
GRIDsearch_Model_One = GridSearchCV(Model_One, PARAM_Model_One, cv=5)
#يتم تدريب النموذج الأفضل (best_model_one) باستخدام البيانات التدريبية (X_train و y_train) ويتم حفظ أفضل الهايبربارامترات (bestparam_model_one).
GRIDsearch_Model_One.fit(X_train, y_train)
BESTparam_Model_One = GRIDsearch_Model_One.best_params_
BEST_Model_One = GRIDsearch_Model_One.best_estimator_




GRIDsearch_Model_Two = GridSearchCV(Model_Two, PARAM_Model_Two, cv=5)
GRIDsearch_Model_Two.fit(X_train, y_train)
BESTparam_Model_Two = GRIDsearch_Model_Two.best_params_
BEST_Model_Two = GRIDsearch_Model_Two.best_estimator_






# تم تدريب وتقييم أفضل النماذج باستخدام التقييم المتقاطع (cross-validation) باستخدام cross_val_score
# وتمريرها أفضل النماذج (best_model1 و best_model2) والبيانات التدريبية (X_train و y_train)
CVscores_Model_One = cross_val_score(BEST_Model_One, X_train, y_train, cv=5)
CVscores_Model_Two = cross_val_score(BEST_Model_Two, X_train, y_train, cv=5)

#  يتم مقارنة النماذج بناءً على تقييم التقسيم المتقاطع، حيث يتم حساب المتوسط ​​لتقييمات التقسيم المتقاطع لكل نموذج
CVscores_Model_One_Mean = CVscores_Model_One.mean()
CVscores_Model_Two_Mean = CVscores_Model_Two.mean()

# يتم طباعة نتائج التقييم المتقاطع والنموذج المختار
print( " من اجل النموذج الاول CROSS_VALIDATION نتائج""\n",CVscores_Model_One)
print( " من اجل النموذج الاول CROSS_VALIDATION نتائج""\n",CVscores_Model_Two)


if CVscores_Model_One_Mean > CVscores_Model_Two_Mean:
    print("النموذج 1 مع Hyperparameters", BESTparam_Model_One, "تم اختياره كأفضل نموذج بناءً على CROSS_VALIDATION درجات.")
    print(" أداؤه العام الأفضل عبر طيات البيانات المختلفة.")
else:
    print("الموديل 2 مع Hyperparameters", BESTparam_Model_Two, "تم اختياره كأفضل نموذج بناءً على CROSS_VALIDATION درجات.")
    print(" أداؤه العام الأفضل عبر طيات البيانات المختلفة.")
    
    
test_data = pd.read_csv("test.csv")
y_pred = BEST_Model_One.predict(test_data)
print(y_pred)
    
    