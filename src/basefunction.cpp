// ���� basefunction.cpp

int inumcore=1; // ��� ������������ ������ ���������� ����
double dterminatedTResudual=1.0e-40; // �������� �������� �� ��������� ������� ����� ����������� ���� ������.
FILE* fp_log;
FILE* fp_statistic_convergence;

int min(int ia, int ib) {
	int ir=ia;
	if (ib<ia) ir=ib;
	return ir;
} // min