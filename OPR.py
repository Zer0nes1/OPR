import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChurnPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Прогнозирование ухода клиентов банка")
        self.root.geometry("900x600")
        
        # Загрузка данных
        self.filepath = r'C:\Users\Илья (-_- )\Documents\OPR\Churn_Modelling.csv'
        self.df, self.customer_ids = self.load_and_prepare_data()
        
        # Обучение модели
        self.model = self.train_model()
        
        # Создание интерфейса
        self.create_widgets()
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        df = pd.read_csv(self.filepath)
        
        # Сохраняем CustomerId для поиска клиентов
        customer_ids = df['CustomerId'].copy()
        
        # Удаление ненужных столбцов
        df = df.drop(['RowNumber', 'Surname', 'EstimatedSalary'], axis=1, errors='ignore')
        
        # Заполнение пропусков
        df['Tenure'] = df['Tenure'].fillna(0).astype('int64')
        
        # Преобразование названий столбцов в нижний регистр
        df.columns = df.columns.str.lower()
        
        return df, customer_ids
    
    def train_model(self):
        """Обучение модели"""
        X = self.df.drop('exited', axis=1)
        y = self.df['exited']
        
        # Определение категориальных и числовых признаков
        categorical_features = ['geography', 'gender']
        numeric_features = ['creditscore', 'age', 'tenure', 'balance', 
                          'numofproducts', 'hascrcard', 'isactivemember']
        
        # Создание пайплайна
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        return model
    
    def predict_single_customer(self):
        """Прогнозирование для одного клиента"""
        try:
            customer_id = int(self.customer_id_entry.get())
            
            if customer_id not in self.customer_ids.values:
                messagebox.showerror("Ошибка", f"Клиент с ID {customer_id} не найден")
                return
            
            # Находим индекс клиента
            idx = self.customer_ids[self.customer_ids == customer_id].index[0]
            
            # Получаем данные клиента
            customer_data = self.df.drop('exited', axis=1).iloc[[idx]]
            
            # Прогнозирование
            proba = self.model.predict_proba(customer_data)[0]
            probability = proba[1] * 100
            
            # Обновляем результаты
            self.result_label.config(text=f"Вероятность ухода: {probability:.2f}%")
            
            if probability > 50:
                self.recommendation_label.config(
                    text="Рекомендация: высокий риск ухода клиента\n\n"
                         "Рекомендуемые действия:\n"
                         "1. Предложить персонального менеджера\n"
                         "2. Сделать специальное предложение\n"
                         "3. Провести опрос о причинах недовольства",
                    fg='red')
            else:
                self.recommendation_label.config(
                    text="Рекомендация: низкий риск ухода клиента\n\n"
                         "Рекомендуемые действия:\n"
                         "1. Поблагодарить за лояльность\n"
                         "2. Предложить программу лояльности\n"
                         "3. Провести кросс-продажу дополнительных услуг",
                    fg='green')
                
            # Показываем данные клиента
            self.show_customer_data(idx)
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректный CustomerId (число)")
    
    def show_customer_data(self, idx):
        """Отображение данных клиента"""
        customer_data = self.df.iloc[idx]
        
        info_text = f"""
        Данные клиента:
        - Возраст: {customer_data['age']}
        - Кредитный рейтинг: {customer_data['creditscore']}
        - Баланс: {customer_data['balance']:.2f}
        - Количество продуктов: {customer_data['numofproducts']}
        - Активный член: {'Да' if customer_data['isactivemember'] else 'Нет'}
        - Страна: {customer_data['geography']}
        - Пол: {customer_data['gender']}
        """
        
        self.customer_info_label.config(text=info_text)
    
    def show_churn_distribution(self):
        """Открытие отдельного окна с гистограммой"""
        # Создаем новое окно
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Распределение уходов клиентов")
        chart_window.geometry("600x500")
        
        # Создаем фигуру matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        
        churn_counts = self.df['exited'].value_counts()
        churn_percentage = churn_counts / churn_counts.sum() * 100
        
        bars = ax.bar(['Остаются', 'Уходят'], churn_percentage, color=['green', 'red'])
        ax.set_title('Распределение клиентов по уходу из банка')
        ax.set_ylabel('Процент клиентов')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Добавляем пояснение под графиком
        explanation = (
            f"Всего клиентов: {len(self.df)}\n"
            f"Остаются: {churn_counts[0]} ({churn_percentage[0]:.1f}%)\n"
            f"Уходят: {churn_counts[1]} ({churn_percentage[1]:.1f}%)"
        )
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Добавляем пояснение
        explanation_label = ttk.Label(chart_window, text=explanation)
        explanation_label.pack(side=tk.BOTTOM, pady=10)
        
        # Кнопка закрытия
        close_btn = ttk.Button(chart_window, text="Закрыть", command=chart_window.destroy)
        close_btn.pack(side=tk.BOTTOM, pady=10)
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Основные фреймы
        input_frame = ttk.LabelFrame(self.root, text="Прогнозирование ухода клиента", padding=10)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_frame = ttk.LabelFrame(self.root, text="Информация и рекомендации", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Элементы для прогнозирования
        ttk.Label(input_frame, text="Введите CustomerId:").pack(pady=5)
        
        self.customer_id_entry = ttk.Entry(input_frame)
        self.customer_id_entry.pack(pady=5)
        
        predict_btn = ttk.Button(input_frame, text="Прогнозировать", command=self.predict_single_customer)
        predict_btn.pack(pady=10)
        
        # Кнопка для показа гистограммы
        chart_btn = ttk.Button(input_frame, text="Показать статистику уходов", 
                              command=self.show_churn_distribution)
        chart_btn.pack(pady=10)
        
        # Результаты прогноза
        self.result_label = ttk.Label(input_frame, text="Вероятность ухода: ", font=('Arial', 12))
        self.result_label.pack(pady=5)
        
        # Информация о клиенте
        self.customer_info_label = ttk.Label(input_frame, text="", justify=tk.LEFT)
        self.customer_info_label.pack(pady=10, fill=tk.X)
        
        # Рекомендации (перенесены в правый фрейм)
        self.recommendation_label = ttk.Label(
            info_frame, 
            text="Здесь будут рекомендации после прогноза", 
            justify=tk.LEFT, 
            wraplength=400
        )
        self.recommendation_label.pack(pady=10, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictorApp(root)
    root.mainloop()