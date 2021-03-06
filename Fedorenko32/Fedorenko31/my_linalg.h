// Файл my_linalg.h :
// собственная реализация некоторых 
// алгоритмов линейной алгебры.
// Версия v0.01 МАРТ 2011
// Затрагивается решение двух проблем линейной алгебры.
// 1. (основная) решение СЛАУ.
// 2. (затронута частично) Проблема нахождения СВ и СЗ.
// ------------------------------------------------------
// Подробнее о решении СЛАУ.
// Рассматриваются: 
// a) плотные и разреженные матрицы.
// b) симметричные и несимметричные.
// Про разреженные матрицы:
// Матрицы предполагаются хорошо обусловленными. (Обязательное диагональное преобладание).
// Для плохо обусловленных матриц, а главным образом с целью
// сокращения количества итераций применяются предобуславливатели.
// Реализовано несколько различных форматов хранения разреженных матриц 
// и функции преобразования между форматами (правда преобразования неэффективны)
// Для быстродействия (в случае когда требуется сразу несколько форматов разреженных 
// матриц ) предпочтение отдаётся хранению матрицы  сразу в нескольких форматах, что
// неэфективно по памяти, но эффективно по быстродействию.

#include "sparse_gauss.c" // метод Гаусса для разреженной матрицы
// Для функций
// calculateSPARSEgaussArray - решает разреженную СЛАУ методом исключения Гаусса.
// initIMatrix - выделение оперативной памяти.
// setValueIMatrix, addValueIMatrix - установка и добавление элемента в разреженную матрицу.

//#define Real double

// Для простейшего хранения разреженной матрицы:
// Ненулевой элемент матрицы СЛАУ:
typedef struct tagNONZEROELEM {
	Real aij;
	int key; // номер столбца
    struct tagNONZEROELEM* next; // указатель на следующий элемент в списке
} NONZEROELEM; 

// Все ненулевые элементы матрицы
typedef struct tagSIMPLESPARSE {
	// массив указателей
   NONZEROELEM** root; // список ненулевых элементов каждой строки
   int n; // число ненулевых элементов
   //int POOL_SIZE; // размер массива ненулевых элементов:
   // n - реальный, POOL_SIZE - размер по памяти.
   //int incCLUSTER_SIZE; // для выделения памяти
} SIMPLESPARSE;

// одна строка матрицы СЛАУ
typedef struct Tequation {
	Real ap, ae, an, aw, as, b;
	int iP, iE, iN, iW, iS;
} equation;


// решает СЛАУ методом исключения Гаусса
// без выбора главного элемента и без 
// учёта разреженности матрицы.
// Подходит для несимметричной матрицы A nodesxnodes.
void eqsolve_simple_gauss(Real **A, int nodes, Real *b, Real *x);

// решает СЛАУ методом разложения Холесского
// Матрица обязательно должна быть симметричной.
// Без учёта разреженности матрицы. Предполагается
// матрица очень хорошая и на диагонали нет нулей.
// По быстродействию в теории в два раза эффективнее 
// метода исключения Гаусса.
// См. книжку Сильвестер и Феррари: МКЭ для радиоэнженеров
// и инженеров электриков. 1986 год.
void eqsolv_simple_holesskii(Real **A, int nodes, Real *b, Real *x);

// Находит обратную матрицу методом
// исключения Гаусса. Приведение к треугольному виду 
// делается только один раз.
// матрица A портится и в неё записывается обратная матрица.
void inverse_matrix_simple(Real** &A, int nodes, bool flag);

// Находит произведение двух квадратных
// матриц nodesxnodes C=A*B. Результат 
// записывается в матрицу B.
void multiply_matrix_simple(Real **A1, Real **A2, int nodes);

// Следующие несколько функций используются как вспомогательные для решения
// полной проблемы собственных значений:

// 1. Умножение квадратных матриц размера nxn:
//              t=m*p.
// По окончании работы результат хранится в матрице t. 
void multi_m(Real **m, Real **p, Real **t, int n);

// 2. Транспонирование квадратной матрицы m размером
// nxn. По окончанию работы результат транспонирования
// хранится в m.
void tr_m(Real **m, int n);

// 3. Возвращает максимальный внедиагональный
// элемент для симметричной матрицы A размером 
// nxn. Позиция максимального элемента A[f][g].
Real max_el(Real **A, int n, int& f, int& g);

// 4. Копирует вторую матрицу в первую: A=B.
// Матрицы квадратные размером nxn
void matr_copy(Real **A1, Real **A2, int n);

// 5. Быстрое умножение двух квадратных матриц специального 
// вида размера nxn (левое умножение):
//                A=hi*A.
// Здесь hi - несимметричная матрица вращения:
// hi[f][f] = cosfi;
// hi[g][g] = cosfi;
// hi[f][g] = +sinfi;
// hi[g][f] = -sinfi;
// Здесь f и g позиции ненулевых элементов.
// Нумерация начинается с нуля.
// По окончании работы результат хранится в исходной матрице A.
// Теперь матрица hiс передаётся только как свои четыре особых элемента
// (cosfi и sinfi), что позволяет существенно экономить память и быстродействие.
// rab - рабочий массив размерности 2xn. Он используется для быстрого умножения.
void multi_m_left(Real **A, Real **rab, int n, int f, int g, Real cosfi, Real sinfi);

// 6. Быстрое умножение двух квадратных матриц специального 
// вида размера nxn (правое умножение):
//                A=A*hi.
// Здесь hi - несимметричная матрица вращения:
// hi[f][f] = cosfi;
// hi[g][g] = cosfi;
// hi[f][g] = -sinfi;
// hi[g][f] = +sinfi;
// Здесь f и g позиции ненулевых элементов.
// Нумерация начинается с нуля.
// По окончании работы результат хранится в исходной матрице A.
// Теперь матрица hi передаётся только как свои четыре особых элемента
// что позволяет существенно экономить память и быстродействие.
// rab - рабочий массив размерности 2xn. Он используется для быстрого умножения.
void multi_m_right(Real **A, Real **rab, int n, int f, int g, Real cosfi, Real sinfi); 
   
// Методом вращений решает полную проблему СЗ в виде
//      A-lambda_scal*E=0
// Процесс нахождения векторов и СЗ является итерационным,
// Его точность характеризуется значением epsilon.
void jacobi_matrix_simple(Real **A, Real **U, Real *lambda, int nodes, Real epsilon);

// Пузырьковая сортировка.
void BubbleSortGSEP1(Real *a, int *mask, int n);

// Первая обобщённая симметричная проблема собственных значений
//   GSEP1: 
void GSEP1(Real **A1, Real **A2, Real **U, Real *lambda, int *mask, int nodes, Real epsilon);

// метод Гаусса для ленточной матрицы A размером
//              nodes x 2*icolx+1, где
//   2*icolx+1 - ширина ленты. Под тем что матрица
//  A ленточная понимается то что ненулевые элементы
//  матрицы содержатся только внутри ленты.
//  b - вектор правой части СЛАУ, x - вектор решение.
//  Нумерация элементов начинается с нуля.
//  Для положительно определённых возможно несимметричных
//  матриц А, которые задаются своей лентой.
//  Гаусс Карл Фридрих 1777-1855.
//  В результате работы матрица А портится.
//  Для треугольных элементов на плоскости и МКЭ
//  ширина ленты может быть больше чем nodes 
//  почти в два раза поэтому использование ленточной технологии 
//  в этом случае неэффективно (бесполезно), если не применять 
//  специальных процедур уменьшения ширины ленты.
void eqsolve_lenta_gauss(Real **A, int nodes, int icolx, Real *b, Real *x);

// Метод (Якоби) Гаусса-Зейделя Ричардсона-Либмана SOR
// для решения СЛАУ с плотной матрицей А nxn
// возможно несимметричной, но с диагональным 
// преобладанием. Матрица А предполагается
// полностью заполненой (неразреженной).
// b - правая часть, x - уточняемое решение, 
// eps - точность определения решения.
// omega - импирически подобранный параметр релаксации.
void Seidel(Real **A, Real *b, Real *x, int n, Real eps, Real omega);

// возвращает максимальное из двух 
// вещественных чисел.
Real fmax(Real fA, Real fB);

// применяется для уравнения поправки давления 
// в случае когда на всей границе стоят условия Неймана.
// Скорость сходимости очень и очень медленная,
// поэтому этот метод используется НИКОГДА.
void SOR(equation* &sl, Real* &x, int n);


// применяется для уравнения поправки давления 
// в случае когда на всей границе стоят условия Неймана.
// Скорость сходимости очень и очень медленная,
// поэтому этот метод используется НИКОГДА.
//void SOR3D(equation3D* &sl, equation3D_bon* &slb, Real* &x, int maxelm, int maxbound, int iVar);

// Метод Сопряжённых градиентов
// без учёта разреженности матрицы СЛАУ.

// умножение матрицы на вектор
Real* MatrixByVector(Real** H,Real* V,int n);

// Евклидова норма вектора
Real NormaV(double *V, int n);

// Скалярное произведение двух векторов
Real Scal(Real *v1, Real *v2, int n);

//----------метод сопряженных градиентов---------------
/* Входные параметры:
*  A - неразреженная матрица СЛАУ,
*  dV - вектор правой части, 
*  x - начальное приближение к решению или NULL.
*  n - размерность СЛАУ Anxn.
*  Матрица A полагается положительно определённой и 
*  симметричной (диагональное преобладание присутствует).
*  Количество итераций ограничено 1000, т.к. предполагается,
*  что если решение не сошлось за 1000 итераций то оно и не сойдётся.
*  Точность выхода по невязке задаётся в глобальной константе:
*  dterminatedTResudual.
*/
Real *SoprGrad(Real **A, Real *dV, Real *x, int n);

// Разряженная симметричная положительно определённая матрица 
// в CRS формате хранения:

// умножение матрицы на вектор
// используется формат хранения CRS
// Разреженная матрица A (val, col_ind, row_ptr) квадратная размером nxn.
// Число уравнений равно числу неизвестных и равно n.
void MatrixCRSByVector(Real* val, int* col_ind, int* row_ptr, Real* V, Real* &tmp, int n);

// умножение транспонированной матрицы на вектор
// (используется, например, в методе BiCG - бисопряжённых градиентов)
// для исходной (не транспонированной матрицы) используется формат хранения CRS
// Разреженная матрица A (val, col_ind, row_ptr) квадратная размером nxn.
// Число уравнений равно числу неизвестных и равно n.
Real* MatrixTransposeCRSByVector(Real* val, int* col_ind, int* row_ptr, Real* V, int n);


/* Входные параметры:
*  val, col_ind, row_ptr - разреженная матрица СЛАУ в формате CRS,
*  dV - вектор правой части, 
*  x - начальное приближение к решению или NULL.
*  n - размерность СЛАУ Anxn.
*  Разреженная матрица A (val, col_ind, row_ptr) квадратная размером nxn.
*  Число уравнений равно числу неизвестных и равно n.
*  Матрица A полагается положительно определённой и 
*  симметричной (диагональное преобладание присутствует).
*  Количество итераций ограничено 1000, т.к. предполагается,
*  что если решение не сошлось за 1000 итераций то оно и не сойдётся.
*  Точность выхода по невязке задаётся в глобальной константе:
*  dterminatedTResudual.
*/
Real *SoprGradCRS(Real *val, int* col_ind, int* row_ptr, Real *dV, Real *x, int n);

// Метод бисопряжённых градиентов
// для возможно несимметричной матрицы А (val, col_ind, row_ptr).
// Запрограммировано по книжке Баландин, Шурина : "Методы
// решения СЛАУ большой размерности".
// dV - правая часть СЛАУ,
// x - начальное приближение к решению или NULL.
// n - размерность А nxn.
// Количество итераций ограничено 2000.
// Точность выхода по невязке задаётся в глобальной константе:
//  dterminatedTResudual.
void BiSoprGradCRS(Real *val, int* col_ind, int* row_ptr, Real *dV, Real* &x, int n, int maxit);

// Прямой ход по разреженной нижнетреугольной матрице L.
// симметричная положительно определённая матрица
// СЛАУ A представлена неполным разложением Холецкого 
// A~=L*transpose(L); L - нижняя треугольная матрица.
// L - хранится в следующем виде:
// 1. ldiag - диагональные элементы L.
// 2. lltr - поддиагональные элементы в строчном формате,
// т.е. хранение построчное. 
// 3. jptr - соотвествующие номера столбцов для lltr, 
// 4. iptr - информация о начале следующей строки для lltr.
// f - вектор правой части размером nodes.
// возвращает вектор z=inverse(L)*f;
// Вектор f портится.
// пример (CSIR - формат):
//  L = 
//  9.0   0.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   11.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   2.0   10.0   0.0   0.0   0.0   0.0   
//  3.0   1.0   2.0   9.0   0.0   0.0   0.0   
//  1.0   0.0   0.0   1.0   12.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  1.0   2.0   0.0   0.0   1.0   0.0   8.0   
// ------------------------------------------
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* inverseL(Real* f, Real* ldiag, Real* lltr, int* jptr, int* iptr, int n);

// Прямой ход по разреженной нижнетреугольной матрице L.
// симметричная положительно определённая матрица
// СЛАУ A представлена неполным разложением Холецкого 
// A~=L*transpose(L); L - нижняя треугольная матрица.
// L - хранится в следующем виде:
// 1. val - диагональные и поддиагональные элементы L.
// в столбцовом порядке. 
// 3. indx - соотвествующие номера строк для val, 
// 4. pntr - информация о начале следующего столбца.
// f - вектор правой части размером nodes.
// возвращает вектор z=inverse(L)*f;
// Вектор f портится.
// пример (CSIR - формат):
//  L = 
//  9.0   0.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   11.0   0.0   0.0   0.0   0.0   0.0   
//  0.0   2.0   10.0   0.0   0.0   0.0   0.0   
//  3.0   1.0   2.0   9.0   0.0   0.0   0.0   
//  1.0   0.0   0.0   1.0   12.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  1.0   2.0   0.0   0.0   1.0   0.0   8.0   
// ------------------------------------------
// val: 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6
// pntr: 0 4 8 10 12 14 15 16
//-------------------------------------------
void inverseL_ITL(Real* f, Real* val, int* indx, int* pntr, Real* &z, int n);

// Обратный ход по разреженной верхнетреугольной матрице U.
// симметричная положительно определённая матрица
// СЛАУ A представлена неполным разложением Холецкого 
// A~=L*transpose(L); L - нижняя треугольная матрица.
// U=transpose(L);
// U - хранится в следующем виде:
// 1. udiag - диагональные элементы U.
// 2. uutr - наддиагональные элементы в столбцовом формате,
// т.е. хранение постолбцовое. 
// Так портрет симметричен, то:
// 3. jptr - соотвествующие номера столбцов для lltr, 
// 4. iptr - информация о начале следующей строки для lltr.
// f - вектор правой части размером nodes.
// возвращает вектор z=inverse(U)*f;
// Вектор f портится.
// пример (CSIR - формат):
//  U=transpose(L) = 
//  9.0   0.0   0.0   3.0   1.0   0.0   1.0   
//  0.0   11.0   2.0   1.0   0.0   0.0   2.0   
//  0.0   0.0   10.0   2.0   0.0   0.0   0.0   
//  0.0   0.0   0.0   9.0   1.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   12.0   0.0   1.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   0.0   8.0   
// ------------------------------------------
// udiag==ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// uutr==lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* inverseU(Real* f, Real* udiag, Real* uutr, int* jptr, int* iptr, int n);

// Обратный ход по разреженной верхнетреугольной матрице U.
// симметричная положительно определённая матрица
// СЛАУ A представлена неполным разложением Холецкого 
// A~=L*transpose(L); L - нижняя треугольная матрица.
// U=transpose(L); - верхняя треугольная матрица.
// U - хранится в следующем виде:
// 1. val - диагональные и наддиагональные элементы U (в строковом формате).
// 2. indx - соотвествующие номера столбцов, 
// 3. pntr - информация о начале следующей строки для val.
// f - вектор правой части размером nodes.
// возвращает вектор z=inverse(U)*f;
// Вектор f портится.
// пример (CSIR_ITL - формат):
//  U=transpose(L) = 
//  9.0   0.0   0.0   3.0   1.0   0.0   1.0   
//  0.0   11.0   2.0   1.0   0.0   0.0   2.0   
//  0.0   0.0   10.0   2.0   0.0   0.0   0.0   
//  0.0   0.0   0.0   9.0   1.0   0.0   0.0   
//  0.0   0.0   0.0   0.0   12.0   0.0   1.0   
//  0.0   0.0   0.0   0.0   0.0   8.0   0.0   
//  0.0   0.0   0.0   0.0   0.0   0.0   8.0 
// ------------------------------------------
// val: 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6
// pntr: 0 4 8 10 12 14 15 16
//-------------------------------------------
void inverseU_ITL(Real* f, Real* val, int* indx, int* pntr, Real* &z, int n);

// Переводит из формата CSIR в формат CSIR_ITL
// форматы:
// CSIR: ldiag, lltr, jptr, iptr
// CSIR_ITL: val, indx, pntr
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
// формат CSIR:
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
//Формируем разреженный формат CSIR_ITL
//val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
//indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
//pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
void convertCSIRtoCSIR_ITL(Real *ldiag, Real *lltr, int *jptr, int *iptr, int n, int nz, Real* &val, int* &indx, int* &pntr, int nnz);

// Неполное разложение Холецкого
// для положительно определённой симметричной
// матрицы А размером nxn.
// n - размерность матрицы СЛАУ
// Матрица val изменяется и в ней возвращается
// неполное разложение Холецкого IC(0).
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
//формат CSIR_ITL (верхний треугольник хранится построчно).
// val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
// indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
// pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
// Результат факторизации без заполнения:
// изменённый массив val (indx и pntr остались без изменений):
// val (factorization)= 
// 3.0
// 1.0
// 0.3333333333333333
// 0.3333333333333333
// 3.3166247903554
// 0.6030226891555273
// 0.30151134457776363
// 0.6030226891555273
// 3.1622776601683795
// 0.6324555320336759
// 2.932575659723036
// 0.34099716973523675
// 3.4472773213410837
// 0.2578524458667825
// 2.8284271247461903
// 2.7310738989293286
//-------------------------------------------
void IC0Factor_ITL(Real* val, int* indx, int* pntr, int n);

// Модифицированное неполное разложение Холецкого.
// (улучшенный вариант IC0Factor_ITL).
void IC0FactorModify_ITL(Real* val, int* indx, int* pntr, int n);

// Переводит из формата CSIR_ITL в формат CSIR (обратное преобразование)
// Память под все массивы предполагается выделенной заранее!!!
// форматы:
// CSIR_ITL: val, indx, pntr
// CSIR: ldiag, lltr, jptr, iptr
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
//Формируем разреженный формат CSIR_ITL
//val : 9.0 3.0 1.0 1.0 11.0 2.0 1.0 2.0 10.0 2.0 9.0 1.0 12.0 1.0 8.0 8.0 
//indx: 0 3 4 6 1 2 3 6 2 3 3 4 4 6 5 6 
//pntr: 0 4 8 10 12 14 15 16 
//--------------------------------------------
// формат CSIR:
// ldiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// lltr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
void convertCSIR_ITLtoCSIR(Real* ldiag, Real* lltr, int* jptr, int* iptr, int n, int nz, Real* val, int* indx, int* pntr, int nnz);

// неполное разложение Холецкого IC(0).
// входные данные нижний треугольник симметричной матрицы в формате CSIR.
// Внутри программы идут преобразования к формату CSIR_ITL библиотеки шаблонов ITL.
void ICFactor0(Real* ldiag, Real* lltr, int* jptr, int* iptr, int n, int nz);

// умножение симметричной положительно определённой  матрицы на вектор 
// используется формат хранения CSIR. В силу симметрии хранятся только поддиагональные элементы altr. 
// Разреженная SPD матрица A (adiag, altr, jptr, iptr) квадратная размером nxn.
// Число уравнений равно числу неизвестных и равно n.
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   2.0   10.0   2.0   0.0   0.0   0.0    
// 3.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 1.0   2.0   0.0   0.0   1.0   0.0   8.0 
// ------------------------------------------
// формат CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0 1.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
void SPDMatrixCSIRByVector(Real* adiag, Real* altr, int* jptr, int* iptr, Real* V, Real* &tmp, int n);

// умножение несимметричной положительно определённой  матрицы на вектор 
// используется формат хранения CSIR.  
// Разреженная матрица A (adiag, altr, autr, jptr, iptr) квадратная размером nxn.
// Число уравнений равно числу неизвестных и равно n.
// Диагональ adiag хранится отдельно. Нижний треугольник altr хранится построчно.
// Верхний треугольник хранится по столбцам autr. Портрет матрицы (позиции ненулевых 
// элементов ) предполагается симметричным. Массив jptr - номера столбцов для нижнего 
// треугольника, массив iptr - показывает где начинаются новые строки для нижнего треугольника.
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   1.0   10.0   2.0   0.0   0.0   0.0    
// 2.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 2.0   2.0   0.0   0.0   3.0   0.0   8.0 
// ------------------------------------------
// формат CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 1.0  2.0 1.0 2.0  1.0 1.0  2.0 2.0 3.0
// autr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* MatrixCSIRByVector(Real* adiag, Real* altr, Real* autr, int* jptr, int* iptr, Real* V, int n);

// умножение транспонированной несимметричной положительно определённой  матрицы на вектор 
// используется формат хранения CSIR.  
// Разреженная матрица A (adiag, altr, autr, jptr, iptr) квадратная размером nxn. Хранится 
// именно исходная матрица, а умножается её транспонированный вариант.
// Число уравнений равно числу неизвестных и равно n.
// Диагональ adiag хранится отдельно. Нижний треугольник altr хранится построчно.
// Верхний треугольник хранится по столбцам autr. Портрет матрицы (позиции ненулевых 
// элементов ) предполагается симметричным. Массив jptr - номера столбцов для нижнего 
// треугольника, массив iptr - показывает где начинаются новые строки для нижнего треугольника.
// пример:
// A = 
// 9.0   0.0   0.0   3.0   1.0   0.0   1.0    
// 0.0   11.0   2.0   1.0   0.0   0.0   2.0    
// 0.0   1.0   10.0   2.0   0.0   0.0   0.0    
// 2.0   1.0   2.0   9.0   1.0   0.0   0.0    
// 1.0   0.0   0.0   1.0   12.0   0.0   1.0    
// 0.0   0.0   0.0   0.0   0.0   8.0   0.0    
// 2.0   2.0   0.0   0.0   3.0   0.0   8.0 
// ------------------------------------------
// формат CSIR:
// adiag: 9.0 11.0 10.0 9.0 12.0 8.0 8.0
// altr: 1.0  2.0 1.0 2.0  1.0 1.0  2.0 2.0 3.0
// autr: 2.0 3.0 1.0 2.0 1.0 1.0 1.0 2.0
// jptr: 1 0 1 2 0 3 0 1 4
// iptr: 0 0 0 1 4 6 6 9
//-------------------------------------------
Real* MatrixTransposeCSIRByVector(Real* adiag, Real* altr, Real* autr, int* jptr, int* iptr, Real* V, int n);


/* Метод сопряжённых градиентов Хестенса и Штифеля [1952]
*  Входные параметры:
*  adiag, altr, jptr, iptr - разреженная матрица СЛАУ в формате CSIR,
*  dV - вектор правой части, 
*  x - начальное приближение к решению или NULL.
*  n - размерность СЛАУ Anxn.
*  nz - размерность массивов altr, jptr.
*  Разреженная матрица A (adiag, altr, jptr, iptr) квадратная размером nxn.
*  Число уравнений равно числу неизвестных и равно n.
*  Матрица A полагается положительно определённой и 
*  симметричной (диагональное преобладание присутствует).
*  Хранится только нижний треугольник с диагональю altr и adiag.
*  Количество итераций ограничено 1000, т.к. предполагается,
*  что если решение не сошлось за 1000 итераций то оно и не сойдётся.
*  Точность выхода по невязке задаётся в глобальной константе:
*  dterminatedTResudual.
*  В качестве предобуславливателя работает неполное разложение Холецкого:
*  M^(-1)==transpose(L)^(-1)*L^(-1); // обращённый предобуславливатель.
*/
Real *SoprGradCSIR(Real* adiag, Real* altr, int* jptr, int* iptr, Real *dV, Real *x, int n, int nz0);

// простая реализация явно преобразующая матрицу СЛАУ А.
// Матрица СЛАУ А задаётся в CSIR формате : adiag, altr, jptr, iptr.
// Неполное разложение Холецкого для А представляет её приближённо в виде:
// A = L*transpose(L); с нулевым заполнением. Массивы jptr и  iptr остаются теми же.
// Тогда матрица : A~=inverse(L)*A*inverse(transpose(L)) тоже симметрична и положительно определена.
// Правая часть преобразованной системы имеет вид: dV~=inverse(L)*dV.
// Решение СЛАУ тогда равно A~*x~=dV~; => x~=transpose(L)*x; => x=inverse(transpose(L))*x~;
// Предобуславливание неполным разлождением Холецкого уменьшает количество итераций при решении СЛАУ,
// улучшает спектральные характеристики матрицы СЛАУ.
Real *SoprGradCSIR2(Real* adiag, Real* altr, int* jptr, int* iptr, Real *dV, Real *x, int n, int nz0);

/* Метод сопряжённых градиентов Хестенса и Штифеля [1952]
*  Входные параметры:
*  M - разреженная матрица СЛАУ в формате SIMPLESPARSE,
*  dV - вектор правой части, 
*  x - начальное приближение к решению или NULL.
*  n - размерность СЛАУ Anxn.
*
*  Разреженная матрица M квадратная размером nxn.
*  Число уравнений равно числу неизвестных и равно n.
*  Матрица M предполагается положительно определённой и 
*  симметричной (диагональное преобладание присутствует).
*  Хранятся только ненулевые элементы. 
*  Количество итераций ограничено 1000, т.к. предполагается,
*  что если решение не сошлось за 1000 итераций то оно и не сойдётся.
*  Точность выхода по невязке задаётся в глобальной константе:
*  dterminatedTResudual.
*  В качестве предобуславливателя работает неполное разложение Холецкого:
*  K^(-1)==transpose(L)^(-1)*L^(-1); // обращённый предобуславливатель.
*  Быстрая реализация (подходит для больших матриц).
*/
void ICCG(SIMPLESPARSE &M, Real *dV, Real* &x, int n, bool bdistwall, int maxiter);

// алгоритм Ю.Г. Соловейчика [1993]
// для возможно несимметричных матриц.
// Запрограммирован по практикуму
// "Численные методы решения систем уравнений" [2004]
// Новосибирского Государственного технического университета.
Real* SoloveichikAlgCSIR_SPD(int isize, // размер квадратной матрицы
						Real* adiag, Real* altr, int* jptr, int* iptr, // матрица СЛАУ
                         Real *dV,  // вектор правой части
                         const Real *dX0, // вектор начального приближения
                         bool bconsole_message); // выводить ли значения невязки на консоль ?

// алгоритм Ю.Г. Соловейчика [1993]
// для возможно несимметричных матриц.
// Здесь используется для симметричной и положительно определённой матрицы.
// С предобуславливанием неполным разложением Холецкого.
// Запрограммирован по практикуму
// "Численные методы решения систем уравнений" [2004]
// Новосибирского Государственного технического университета.
Real* SoloveichikAlgCSIR_SPDgood(int isize, int nz0,// размер квадратной матрицы
						Real* adiag, Real* altr, int* jptr, int* iptr, // матрица СЛАУ
                         Real *dV,  // вектор правой части
                         const Real *dX0, // вектор начального приближения
                         bool bconsole_message); // выводить ли значения невязки на консоль ?

// алгоритм Ю.Г. Соловейчика [1993]
// для возможно несимметричных матриц.
// Запрограммирован по практикуму
// "Численные методы решения систем уравнений" [2004]
// Новосибирского Государственного технического университета.
void SoloveichikAlgCRS(int isize, // размер квадратной матрицы
						Real *val, int* col_ind, int* row_ptr, // матрица СЛАУ
                         Real *dV,  // вектор правой части
                         const Real* &dX0, // вектор начального приближения
                         bool bconsole_message, int maxit); // выводить ли значения невязки на консоль ?

// инициализирует разреженную матрицу
void initsimplesparse(SIMPLESPARSE &M, int nodes);

// Добавляет ненулевой элемент в
// простейшую разряженную матрицу M
void addelmsimplesparse(SIMPLESPARSE &M, Real aij, int i, int j, bool bset);

// освобождение памяти для матрицы SIMPLESPARSE
void simplesparsefree(SIMPLESPARSE &M, int nodes);

// Для генерации матрицы СЛАУ требуется переупорядочивание элементов
// сортировка. Здесь будет реализована быстрая сортировка.
// Брайан Керниган и Денис Ритчи "The C programming language".
// swap: Обмен местами v[i] и v[j]
void swap(int* &v, int i, int j);

// Вот алгоритм PivotList
int PivotList(int* &list, int first, int last);

// Быстрая сортировка Хоара.
// Запрограммировано с использованием ДЖ. Макконелл Анализ алгоритмов
// стр. 106.
void QuickSort(int* &list, int first, int last);

// Преобразует простейший формат хранения разреженной матрицы
// в формат CRS. Всего nodes - уравнений.
void simplesparsetoCRS(SIMPLESPARSE &M, Real* &val, int* &col_ind, int* &row_ptr, int nodes);

// Преобразует equation3D  формат хранения в CRS формат.
// Цель написания этого преобразователя: экономия оперативной памяти компьютера.
// Т.к. формат SIMPLESPARSE требует слишком много памяти.
//void equation3DtoCRS(equation3D* &sl, equation3D_bon* &slb, Real* &val, int* &col_ind, int* &row_ptr, int maxelm, int maxbound, Real alpharelax);

// Реализация на связном списке.
// Преобразует простейший формат хранения разреженной матрицы
// в формат CSIR. Всего nodes - уравнений.
// Это работает только для SPD матриц.
// Симметричный положительно определённый случай,
// хранится только нижний треугольник.
void simplesparsetoCSIR(SIMPLESPARSE &M, Real* &adiag, Real* &altr, int* &jptr, int* &iptr, int nodes);

// печать матрицы в консоль
void printM_and_CSIR(SIMPLESPARSE &sparseM, int  n);

// Реализация на связном списке.
// Преобразует простейший формат хранения разреженной матрицы
// в формат CSIR_ITL. Всего nodes - уравнений.
// Это работает только для SPD матриц.
// Симметричный положительно определённый случай,
// хранится только верхний треугольник.
// Память выделяется внутри метода.
void simplesparsetoCSIR_ITLSPD(SIMPLESPARSE &M, Real* &val, int* &indx, int* &pntr, int nodes);

/* Неполное LU разложение для несимметричных матриц
*  Пример А nxn=
*    9.0 0.0 0.0 3.0 1.0 0.0 1.0
*    0.0 11.0 2.0 1.0 0.0 0.0 2.0 
*    0.0 1.0 10.0 2.0 0.0 0.0 0.0 
*    2.0 1.0 2.0 9.0 1.0 0.0 0.0 
*    1.0 0.0 0.0 1.0 12.0 0.0 1.0 
*    0.0 0.0 0.0 0.0 0.0 8.0 0.0
*    2.0  2.0 0.0 0.0 3.0 0.0 8.0
*-----------------------------------------
*  инициализация (в этом виде данные поступают на вход процедуре):
*  верхняя треугольная матрица хранится построчно, в каждой строке
*  элементы отсортированы по убыванию номеров столбцов.
*  U_val :   1.0, 1.0, 3.0, 9.0,   2.0, 1.0, 2.0, 11.0,   2.0, 10.0, 1.0, 9.0, 1.0,12.0, 8.0, 8.0
*  U_ind :   6, 4, 3, 0,  6, 3, 2, 1,  3,2, 4,3, 6,4, 5, 6
*  U_ptr :   0, 4, 8, 10, 12, 14, 15, 16
*  нижняя треугольная матрица хранится постолбцово, в каждом столбце
*  элементы отсортированы по убыванию номеров строк.
*  L_val :  2.0, 1.0, 2.0, 9.0,    2.0, 1.0, 1.0, 11.0,  2.0, 10.0, 1.0, 9.0,  3.0, 12.0, 8.0, 8.0
*  L_ind :  6, 4, 3, 0,  6, 3, 2, 1,   3, 2,  4,3,  6, 4, 5, 6
*  L_ptr :  0, 4, 8, 10, 12, 14, 15, 16
*----------------------------------------------
* Результат ILU разложения:
* U_val : 1.0, 1.0, 3.0, 9.0, 2.0, 1.0, 2.0, 11.0, 2.0, 10.0, 1.0, 9.0, 1.0, 12.0, 8.0, 8.0.
* L_val : 0.222, 0.111, 0.222, 1.0, -1.273, 0.091, 0.091, 1.0, 0.2, 1.0, 0.111, 1.0, -0.417, 1.0, 1.0, 1.0.
*/
void ILU0_Decomp_ITL(Real* &U_val, int* &U_ind, int* &U_ptr, Real* &L_val, int* &L_ind, int* &L_ptr, int n);

// неполное LU разложение с нулевым заполнением из книги Й. Саада
// Iterative Methods for Sparse linear systems.
// на вход подаётся матрица А в CRS формате.
// на выходе матрица luval в CRS формате в матрице L на диагонали 1.0,
// uptr - указатели на диагональные элементы.
//void ilu0_Saadtest(); // проверено
void ilu0_Saad(int n, Real* a, int* ja, int* ia, Real* &luval, int* &uptr, int &icode);

/* Метод бисопряжённых градиентов
* для возможно несимметричной матрицы А (val, col_ind, row_ptr).
* Запрограммировано по книжке Баландин, Шурина : "Методы
* решения СЛАУ большой размерности".
* dV - правая часть СЛАУ,
* x - начальное приближение к решению или NULL.
* n - размерность А nxn.
* Количество итераций ограничено maxiter.
* Точность выхода по невязке задаётся в глобальной константе:
*  dterminatedTResudual.
* Иногда метод расходится. Если выбрать другой вектор r_tilda, то 
* процесс может стать сходящимся. Ограничение на выбор вектора r_tilda:
* главное чтобы скалярное произведение Scal(r,r_tilda,n) != 0.0.
*//*
void BiSoprGrad(IMatrix *xO, equation3D* &sl, equation3D_bon* &slb, Real *dV, Real* &x, int maxelm, int maxbound, bool bSaad, Real alpharelax, int  maxiter);
*/

// алгоритм Ю.Г. Соловейчика [1993]
// для возможно несимметричных матриц.
// Запрограммирован по практикуму
// "Численные методы решения систем уравнений" [2004]
// Новосибирского Государственного технического университета.
// Добавлен ILU0 предобуславливатель.
/*
void SoloveichikAlg( IMatrix *xO, equation3D* &sl, equation3D_bon* &slb,// Разреженная матрица СЛАУ
					     int maxelm, int maxbound, // число внутренних и граничных КО
                         Real *dV,  // вектор правой части
                         Real* &dX0, // вектор начального приближения
                         bool bconsole_message, // выводить ли значения невязки на консоль ?
						 bool bSaad, // если bSaad==true то использовать ilu0 разложение из книги Й. Саада иначе использовать ITL ilu0 разложение. 
						 int imaxiter, Real alpharelax); // максимально допустимое кол-во итераций
						 */

// Метод Ван Дер Ворста Bi-CGStab
void Bi_CGStabCRS(int n, Real *val, int* col_ind, int* row_ptr, Real *dV, Real* &dX0, int maxit);

// Метод Ван Дер Ворста Bi-CGStab
// работает для возможно несимметричных вещественных матриц.
// встроен предобуславливатель ILU(0).
// Метод является комбинацией методов BiCG и GMRES(1). 
/*void Bi_CGStab(IMatrix *xO, equation3D* &sl, equation3D_bon* &slb,
			   int maxelm, int maxbound,
			   Real *dV, Real* &dX0, int maxit, Real alpharelax);
			   */

// А.А.Фомин, Л.Н.Фомина 
// Ускорение полилинейного рекуррентного метода в подпространствах крылова.
// Вестник томского государственного университета. Математика и механика №2(14) 2011год.
// Алгоритм основан на прямом сочетании алгоритмов LR1 и Bi-CGStab P.
// LR1 - полилинейный метод предложенный еще в книге С. Патанкара : гибрид
// прямого метода прогонки (алгоритм Томаса) и метода Гаусса-Зейделя.
// Bi-CGStab P - алгоритм Ван Дер Ворста с предобуславливанием : гибрид Bi-CG и GMRES(1).
// начало написания, тестирования и использования в AliceFlow_v0_06 датируется 
// 24 октября 2011 года на основе предыдущих разработок.
/*
void LR1sK(FLOW &f, equation3D* &sl, equation3D_bon* &slb,
	       Real *val, int* col_ind, int* row_ptr,
		   int maxelm, int maxbound, int iVar,
		   Real *dV, Real* &dX0, int maxit, bool bprintmessage, bool bexporttecplot);
		   */
