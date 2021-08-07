// файл sparse_gauss.h 
// Реализация метода Гаусса для разреженной матрицы.

#define Real double

// тип элемент
typedef struct tagTERM{
	int key;
	Real val;
	// специальные поля для
	// АВЛ дерева, при реализации
	// интерфейса строк и стобцов
	// на массиве они не используются.
	struct tagTERM *left;
	struct tagTERM *right;
	int bal;
} TERM;

// строка или столбец в матрице СЛАУ
typedef struct tagIRow{
    Real eps0; // для определения вещественного нуля

	// следующие два поля используются
	// только при реализации интерфейса 
	// строк и столбцов на динамическом массиве.
	int POOL_SIZE; // размер массива включая нулевые элементы
	int n; // число ненулевых элементов

	TERM *elm;

} IRow;

// разреженная матрица СЛАУ:
// Квадратная nxn с диагональным преобладанием,
// возможно несимметричная. Нумерация начинается 
// с нуля.
typedef struct tagIMatrix{
    Real eps0; // для определения вещественного нуля

	int n; // размерность матрицы nxn.
    // jp - строки верхней полуматрицы
    // jm - столбцы нижней полуматрицы
    // dd - главная диагональ
	IRow *jp, *jm;
	Real *dd;
} IMatrix;

// поиск ячейки с индексом key  
int search_i(TERM* list, int n, int key);

// удаляет элемент с ключём равным key
void deleteAt(TERM* &list, int key, int &n, int &pool);

// добаляет элемент со значениями : num, val.
void add(TERM* &list, int &n, int &pool, int num, Real val);

// возвращает значение ячейки ключ
// которой равен key
Real get_val(TERM* list, int n, int key);

// добавляет число value в ячейку
// с ключём key
void modify_add(TERM* &list, int n, int key, Real value);

// устанавливает число value в ячейку
// с ключём key
void modify_set(TERM* &list, int n, int key, Real value);

// зависит от внутреннего представления
// Возвращает все ненулевые элементы
void get_values(TERM *list, int n, int* &indexes, Real* &values);

// возвращает 0 при i<=0, i при i>0
int sigma(int i);

// преобразование row,col в координаты полуматриц (d,j)
// для верхней полуматрицы: d=0..(n-1), j=-(n-d-1)..-1
// для нижней полуматрицы:  d=0..(n-1), j=1..(n-d-1)
int getD(int row, int col);
int getJ(int row, int col);

// обратное преобразование координат полуматриц (d,j) в row,col
int getRow(int d, int j);
int getCol(int d, int j);


// записывает значение value в ячейку с индексом num
void setValueIRow(IRow *xO, int num, Real value);

// добавляет value к существующему значению в ячейке num
void addValueIRow(IRow *xO, int num, double value);

// возвращает значение ячейки num
Real getValueIRow(IRow *xO, int num);

// возвращает все ненулевые ячейки строки/столбца: 
// индексы ячеек - в indexes, значения в values
int getValuesIRow(IRow *xO, int* &indexes, Real* &values);

// выделение памяти под разреженную матрицу
void initIMatrix(IMatrix *xO, int n);

// освобождение памяти из под объекта
void freeIMatrix(IMatrix* xO);

// устанавливает значение value в ячейку с координатами [row,col];
// row - номер строки матрицы
// col - номер столбца матрицы
void setValueIMatrix(IMatrix *xO, int row, int col, Real value);

// добавляет значение value к ячейке [row,col]
void addValueIMatrix(IMatrix *xO, int row, int col, double value);

// возвращает значение ячейки [row,col]
Real  getValueIMatrix(IMatrix *xO, int  row, int  col);

// возвращает ненулевые значения и индексы ячеек строки d,
// которые находятся правее главной диагонали
int  getJRowIMatrix(IMatrix *xO, int  d, int* &indexes, Real* &values);

// возвращает ненулевые значения и индексы ячеек столбца d, 
// которые находятся ниже главной диагонали
int  getJColIMatrix(IMatrix *xO, int  d, int* &indexes, Real* &values);

// главный метод, возвращающий решение x,
// принимает вектор свободных членов b и 
// квадратную матрицу xO в специальном разреженном формате.
void calculateSPARSEgaussArray(IMatrix *xO, Real *x, Real *b);

// Преобразует матрицу в формате IMatrix в CSIR формат 
// совместимый с библиотекой ITL для реализации ILU разложения.
void convertIMatrixtoCSIR_ILU_ITL(IMatrix *xO, Real* &U_val, int* &U_ind, int* &U_ptr, Real* &L_val, int* &L_ind, int* &L_ptr);

