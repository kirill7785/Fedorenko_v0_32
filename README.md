# Fedorenko_v0_32
2D solver in rectangle

2D решатель в прямоугольнике для системы уравнений Навье-Стокса в приближении Обербека-Буссинеска
 в переменных вихрь-функция тока-температура на основе:
 1. дискретного преобразования Фурье.
 2. алгебраического многосеточного метода amg1r5.  Данный метод не имеет ограничений на форму расчётной области.
 3. геометрического двух (трёх) сеточного метода Р.П. Федоренко (его статья 1961). 
 4. динамического программирования Р.Беллман и Энджел "Динамическое программирование и уравнения в частных производных".
 5. метода ILU разложения для ленточной матрицы. Обратная матрица находится один раз, а для решения уравнения Пуасона используется обратный ход по верхнетреугольной матрице за O(N) операций. Данный метод не имеет ограничений на форму расчётной области.
 6. метода прогонки как компонента метода расщепления по координатным направлениям (Писмен и Речфорд).

## Примеры расчётов

Релей Бенар
Число Прандтля Pr=0.67; Число Грасгофа 142.9.
Расчётная сетка 350 на 50 улов. Время расчёта 1мин 34с на 2*intel xeon 2630v4.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Рэлей-Бенар/Расчётная%20сетка.png)
Расчётная сетка.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Рэлей-Бенар/Скорость.png)
Модуль скорости жидкости.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Рэлей-Бенар/Температура.png)
Температура жидкости.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Рэлей-Бенар/Функция%20тока.png)
Функция тока.

Течение расплава в печи Чохральского
Число Прандтля Pr=0.67; Число Грасгофа 142.9.
Расчётная сетка 50 на 50 улов. Время расчёта 7с на 2*intel xeon 2630v4.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Чохральский/Расчётная%20сетка.png)
Расчётная сетка.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Чохральский/Модуль%20скорости%20Чохральский.png)
Модуль скорости жидкости.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Чохральский/Температура%20Чохральский.png)
Температура жидкости.
![alt_text](https://github.com/kirill7785/Fedorenko_v0_32/blob/main/pic/Чохральский/Функция%20тока%20Чохральский.png)
Функция тока.
