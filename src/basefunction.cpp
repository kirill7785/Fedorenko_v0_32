// Файл basefunction.cpp

int inumcore=1; // для параллельной версии количество ядер
double dterminatedTResudual=1.0e-40; // финишная точность по истечении которой метод заканчивает свою работу.
FILE* fp_log;
FILE* fp_statistic_convergence;

int min(int ia, int ib) {
	int ir=ia;
	if (ib<ia) ir=ib;
	return ir;
} // min