/* Метод Гаусса для разреженной матрицы на массиве.
* 6 марта 2011.
*/
//#include "sparse_gauss.h" // объявление всех функций
// объявление функций и реализация интерфейса строк и столбцов
#include "irow_realise_array.c" // на массиве


// возвращает 0 при i<=0, i при i>0
int sigma(int i) {
	int ir=0;
	if (i>0) ir=i;

	return ir;
} // sigma

// преобразование row,col в координаты полуматриц (d,j)
// для верхней полуматрицы: d=0..(n-1), j=-(n-d-1)..-1
// для нижней полуматрицы:  d=0..(n-1), j=1..(n-d-1)
int getD(int row, int col)
{
    return row-sigma(row-col);
}
int getJ(int row, int col)
{
    return col-row;
}
// обратное преобразование координат полуматриц (d,j) в row,col
int getRow(int d, int j)
{
    return d + sigma(-j);
}
int getCol(int d, int j)
{
    return d + sigma(j);
}

// записывает значение value в ячейку с индексом num
void setValueIRow(IRow *xO, int num, Real value) {

	// int i = indexes.IndexOf(num);
	int i=-1;
	// поиск ячейки с индексом num
	i=search_i(xO->elm, xO->n, num); 

    // если записывается 0-вое значение, то удаляем данную ячейку
    if (fabs(value)<xO->eps0)
    {
       if (i!=-1)
       {
           //indexes.RemoveAt(i);
           //values.RemoveAt(i);

		   deleteAt(xO->elm,num,xO->n,xO->POOL_SIZE); // удаление элемента с ключём i

       }
      
     }
	  else 
	 {

        // если значение не 0-вое, то перезаписываем или добавляем ячейку
        if (i!=-1)
        {
			modify_set(xO->elm, xO->n, num, value);
        }
          else
        {
            //indexes.Add(num);
            //values.Add(value);

			add(xO->elm, xO->n, xO->POOL_SIZE, num, value);          

        }
	 }

} // setValueIRow

// добавляет value к существующему значению в ячейке num
void addValueIRow(IRow *xO, int num, double value)
{
    // int i = indexes.IndexOf(num);
	int i=-1;
	// поиск ячейки с индексом num
	i=search_i(xO->elm, xO->n, num);

    if (i!=-1)
    {
		modify_add(xO->elm, xO->n, num, value);
    }
     else
    {
       //indexes.Add(num);
       //values.Add(value);

		add(xO->elm, xO->n, xO->POOL_SIZE, num, value);

    }
} // addValueIRow

// возвращает значение ячейки num
// требуется два линейных поиска.
Real getValueIRow(IRow *xO, int num)
{
   // int i = indexes.IndexOf(num);
   int i=-1;
   // поиск ячейки с индексом num
   i=search_i(xO->elm, xO->n, num);

   if (i!=-1) return (Real) get_val(xO->elm, xO->n, num); 
   return 0.0;
} // getValueIRow

// возвращает все ненулевые ячейки строки/столбца: 
// индексы ячеек - в indexes, значения в values
int getValuesIRow(IRow *xO, int* &indexes, Real* &values)
{
	if (xO->n>0) {
		indexes = new int[xO->n];
	    values = new Real[xO->n];

	    get_values(xO->elm, xO->n, indexes, values);
	}
	return xO->n;

} // getValuesIRow

// выделение памяти под разреженную матрицу
void initIMatrix(IMatrix *xO, int n) {
	if (xO == NULL) xO=new IMatrix;
	xO->eps0=1e-100; // для отделения вещественного нуля
	xO->n=n;
	xO->dd=new Real[n];
	xO->jp=new IRow[n];
	xO->jm=new IRow[n];
	int i1; // счётчик цикла for
	for (i1=0; i1<n; i1++) {
		xO->dd[i1]=0.0;
		xO->jp[i1].n=0;
		xO->jm[i1].n=0;
		xO->jp[i1].elm=NULL;
		xO->jm[i1].elm=NULL;
		xO->jp[i1].POOL_SIZE=0;
		xO->jm[i1].POOL_SIZE=0;
		xO->jp[i1].eps0=xO->eps0;
		xO->jm[i1].eps0=xO->eps0;
	}
} // initIMatrix

// освобождение памяти из под объекта
void freeIMatrix(IMatrix* xO) {
	// для того чтобы избежать ошибок при повтоном
	// уделении одного и того же объекта здесь сделана
	// проверка и если xO->n==0 то объект уже удалён.
	if (xO->n!=0) 
	{
		if (xO->dd!=NULL) delete xO->dd;
		xO->dd=NULL;
	    int i=0;
	    for (i=0; i<xO->n; i++) {
		    delete xO->jp[i].elm;
		    delete xO->jm[i].elm;
	    }
	    if (xO->jp!=NULL) delete xO->jp;
	    if (xO->jm!=NULL) delete xO->jm;
		xO->jp=NULL;
	    xO->jm=NULL;
	    xO->n=0;
	}
} // freeIMatrix

// устанавливает значение value в ячейку с координатами [row,col];
// row - номер строки матрицы
// col - номер столбца матрицы
void setValueIMatrix(IMatrix *xO, int row, int col, Real value)
{
    if (row==col)
    {
		xO->dd[row] = value;
    }
	  else
	{
       int d = getD(row,col);
       int j = getJ(row,col);
	   if (j>0) setValueIRow(&(xO->jp[d]), j, value);
	     else setValueIRow(&(xO->jm[d]), -j, value); 
	}
} // setValueIMatrix

// добавляет значение value к ячейке [row,col]
void addValueIMatrix(IMatrix *xO, int row, int col, double value)
{
	// Если добавляемое значение ненулевое
	if (fabs(value)>xO->eps0) {
		if (row==col)
        {
			xO->dd[row] += value;
        }
		else
		{
           int  d = getD(row,col);
           int  j = getJ(row,col);
		   if  (j>0) addValueIRow(&(xO->jp[d]), j, value);
		      else addValueIRow(&(xO->jm[d]), -j, value); 
		}
	}
} // addValueIMatrix

// возвращает значение ячейки [row,col]
Real  getValueIMatrix(IMatrix *xO, int  row, int  col)
{
   Real ret; // Возвращаемое значение
   if  (row==col) ret=xO->dd[row];
   else {
	   int  d = getD(row,col);
       int  j = getJ(row,col);
	   if (j>0) {
		   ret=getValueIRow(&(xO->jp[d]), j);
	   }
	   else
	   {
		   ret=getValueIRow(&(xO->jm[d]), -j);
	   }
   }
   return ret;
} //getValueIMatrix 

// возвращает ненулевые значения и индексы ячеек строки d,
// которые находятся правее главной диагонали
int  getJRowIMatrix(IMatrix *xO, int  d, int* &indexes, Real* &values)
{
    int in=0; // количество ненулевых элементов
	in=getValuesIRow(&(xO->jp[d]), indexes, values);
    for  (int  i=0; i<in; i++) indexes[i] = getCol(d,indexes[i]);
	return in;
} // getJRowIMatrix

// возвращает ненулевые значения и индексы ячеек столбца d, 
// которые находятся ниже главной диагонали
int  getJColIMatrix(IMatrix *xO, int  d, int* &indexes, Real* &values)
{
    int in=0; // количество ненулевых элементов
    in=getValuesIRow(&(xO->jm[d]), indexes, values);
    for  (int  i=0; i<in; i++) indexes[i] = getRow(d,-indexes[i]);
	return in;
} // getJColIMatrix

// главный метод, возвращающий решение x,
// принимает вектор свободных членов b и 
// квадратную матрицу xO в специальном разреженном формате.
// реализация без барьера и итерационного уточнения.
void calculateSPARSEgaussArray(IMatrix *xO, Real *x, Real *b) {
    
	// col - столбец, row - строка

	// Все ненулевые значения обнуляемого столбца
	int * colIndexes=NULL;
	Real * colValues=NULL;

    // Ненулевые ячейки строки правее главной диагонали
	int * rowIndexes=NULL;
	Real * rowValues=NULL;
    
    int colIndexesLength, rowIndexesLength;

	Real dd; // диагональный элемент
	Real M;

	// приведение к верхнетреугольному виду
	for (int col=0; col<xO->n-1; col++) 
	{
        // получаем все ненулевые значения обнуляемого столбца
        colIndexesLength=getJColIMatrix(xO, col, colIndexes, colValues);
        // получаем индексы и значения ячеек строки, правее главной диагонали
		rowIndexesLength=getJRowIMatrix(xO, col, rowIndexes, rowValues);

        // получаем элемент главной диагонали, которым будем обнулять столбец
        dd = getValueIMatrix(xO,col,col);

		for (int i=0; i<colIndexesLength; i++) {
            M = colValues[i]/dd;

			// M подобрано таким образом чтобы обнулить ячейку столбца
			setValueIMatrix(xO,colIndexes[i],col,0.0);

           
			// складываем строки
			for (int ii=0; ii<rowIndexesLength; ii++) {
				// -M*A[k][j] появление нового ненулевого элемента 
				addValueIMatrix(xO, colIndexes[i], rowIndexes[ii],-M*rowValues[ii]);
			}
             
            // складываем соответствующие свободные члены
            b[colIndexes[i]] -= M*b[col];
		}
	}

	Real sum; // сумматор

    // используя обратный ход находим неизвестные
    for  (int  row = xO->n-1; row>=0; row--)
    {
       sum = 0.0;
       // получаем индексы и значения ячеек строки, правее главной диагонали
       rowIndexesLength=getJRowIMatrix(xO, row, rowIndexes, rowValues);
       for  (int  i=0; i<rowIndexesLength; i++) sum += x[rowIndexes[i]]*rowValues[i];
	   // получаем элемент главной диагонали, которым будем обнулять столбец
       dd = getValueIMatrix(xO,row,row);
       x[row] = (b[row]-sum)/dd;
    }

} // calculateSPARSEgaussArray

// Преобразует из формата CRS в формат IMatrix.
void convertCRStoIMatrix(int n, Real* luval, int* ja, int* ia, int* uptr, IMatrix* sparseS) {

	//printf("commin\n");
    
	int i=0, j=0;
	for (i=0; i<n; i++) {
		for (j=ia[i]; j<ia[i+1]; j++) {
			//printf("%d %d\n",i,ja[j]);
			setValueIMatrix(sparseS, i, ja[j], luval[j]);
		}
	}
	//printf("commin\n");

} // convertCRStoIMatrix

// Преобразует матрицу в формате IMatrix в CSIR формат 
// совместимый с библиотекой ITL для реализации ILU разложения.
// Оперативная память выделяется внутри.
void convertIMatrixtoCSIR_ILU_ITL(IMatrix *xO, Real* &U_val, int* &U_ind, int* &U_ptr, Real* &L_val, int* &L_ind, int* &L_ptr) {
	int n, nz;
	n=xO->n; // размерность квадратной матрицы
	int i,j; // счётчики цикла for

	// Верхняя треугольная матрица.
    nz=n;
	
	for (i=0; i<n; i++) nz+=xO->jp[i].n; // число ненулевых элементов в верхней треугольной матрице
	U_val = new Real[nz]; // диагональные и наддиагональные элементы.
	U_ind = new int[nz];
	U_ptr = new int[n+1];
	for (i=0; i<nz; i++) {
		U_val[i]=0.0;
		U_ind[i]=0;
	}
	for (i=0; i<=n; i++) U_ptr[i]=nz;

    // Ненулевые ячейки строки правее главной диагонали
	int * rowIndexes=NULL;
	Real * rowValues=NULL;

	int colIndexesLength, rowIndexesLength;

    int ik=0; // счётчик ненулевых наддиагональных элементов СЛАУ

	// По всем строкам вниз кроме последней
	for (i=0; i<n-1; i++) {
		// получаем индексы и значения ячеек строки, правее главной диагонали
        rowIndexesLength=getJRowIMatrix(xO, i, rowIndexes, rowValues);
		// BubbleSort по убыванию.
		for (int i1=1; i1<rowIndexesLength; i1++)
			for (int j1=rowIndexesLength-1; j1>=i1; j1--) 
				if (rowIndexes[j1-1]<rowIndexes[j1]) {
					Real rtemp=rowValues[j1-1];
                    rowValues[j1-1]=rowValues[j1];
                    rowValues[j1]=rtemp;
					int itemp=rowIndexes[j1-1];
					rowIndexes[j1-1]=rowIndexes[j1];
					rowIndexes[j1]=itemp;
				}

		for (j=0; j<rowIndexesLength; j++) {
			U_val[ik]=rowValues[j]; // ненулевое значение
			U_ind[ik]=rowIndexes[j]; // номер столбца
			U_ptr[i]=min(ik,U_ptr[i]);
			ik++;
		}
		// диагональный элемент
		U_val[ik]=xO->dd[i];
        U_ind[ik]=i;
        U_ptr[i]=min(ik,U_ptr[i]);
		ik++;

		// освобождение оперативной памяти
		if (rowIndexesLength>0) {
			delete rowIndexes; 
	        delete rowValues;
		}
	}
    // Добавление последнего диагонального элемента
	U_val[ik]=xO->dd[n-1];
    U_ind[ik]=n-1;
    U_ptr[n-1]=min(ik,U_ptr[n-1]);
	ik++;
    
	// Сортировки элементов не производится!


	// Нижняя треугольная матрица:
    nz=n;

	// число ненулевых элементов в нижней треугольной матрице
	for (i=0; i<n; i++) nz+=xO->jm[i].n; 
	L_val = new Real[nz];
	L_ind = new int[nz];
	L_ptr = new int[n+1];
	for (i=0; i<nz; i++) {
		L_val[i]=0.0;
		L_ind[i]=0;
	}
	for (i=0; i<=n; i++) L_ptr[i]=nz;

    // Все ненулевые значения в столбце ниже главной диагонали
	int * colIndexes=NULL;
	Real * colValues=NULL; 

    ik=0; // счётчик ненулевых наддиагональных элементов СЛАУ

	// По всем столбцам вправо кроме последнего
	for (i=0; i<n-1; i++) {
		// получаем индексы и значения ячеек столбца, ниже главной диагонали
		colIndexesLength=getJColIMatrix(xO, i, colIndexes, colValues);
		// BubbleSort по убыванию.
		for (int i1=1; i1<colIndexesLength; i1++)
			for (int j1=colIndexesLength-1; j1>=i1; j1--) 
				if (colIndexes[j1-1]<colIndexes[j1]) {
					Real rtemp=colValues[j1-1];
                    colValues[j1-1]=colValues[j1];
                    colValues[j1]=rtemp;
					int itemp=colIndexes[j1-1];
					colIndexes[j1-1]=colIndexes[j1];
					colIndexes[j1]=itemp;
				}

		for (j=0; j<colIndexesLength; j++) {
			L_val[ik]=colValues[j]; // ненулевое значение
			L_ind[ik]=colIndexes[j]; // номер столбца
			L_ptr[i]=min(ik,L_ptr[i]);
			ik++;
		}
		// диагональный элемент
		L_val[ik]=xO->dd[i];
        L_ind[ik]=i;
        L_ptr[i]=min(ik,U_ptr[i]);
		ik++;

		// освобождение оперативной памяти
		if (colIndexesLength>0) {
			delete colIndexes;
	        delete colValues;
		}

	}
    // Добавление последнего диагонального элемента
	L_val[ik]=xO->dd[n-1];
    L_ind[ik]=n-1;
    L_ptr[n-1]=min(ik,U_ptr[n-1]);
	ik++;

	// Сортировки элементов не производится!


} // convertIMatrixtoCSIR_ILU_ITL