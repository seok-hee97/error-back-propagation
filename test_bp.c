#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>

#define InputUnitNo 2
#define HiddenUnitNo 3
#define OutputUnitNo 1
#define MaxPatternNo 4

#define Eta 0.75
#define Alpha 0.25
#define ErrorFunc 0.05
#define Wmin -0.30	//����ġ�� �ʱⰪ ����
#define Wmax 0.30		//����ġ�� �ʱⰪ ����

#define f(x) (1/(1+exp(-x))) //�ñ׸��̵� �Լ�
#define rnd() ((double)rand()/0x7fff * (Wmax - Wmin)+Wmin) //0x7fff * (Wmax - Wmin) + Wmin) -> ������ ����

//��Ʈ��ũ ����: �Է��� 2���̴ϱ� 2���� �迭 ���
double O1[MaxPatternNo][InputUnitNo];
double O2[HiddenUnitNo];
double O3[OutputUnitNo];

double t[MaxPatternNo][OutputUnitNo];
double W21[HiddenUnitNo][InputUnitNo];
double dW21[HiddenUnitNo][InputUnitNo];
double W32[OutputUnitNo][HiddenUnitNo];
double dW32[OutputUnitNo][HiddenUnitNo];

double bias2[HiddenUnitNo];
double dbias2[HiddenUnitNo];
double bias3[OutputUnitNo];
double dbias3[OutputUnitNo];

int learning_pattern_no;
int test_pattern_no;


int main() {
	int i, j, k;
	char filename[30];
	char ss[80];
	double errorfunc;

 	void propagation(int p);
	void back_propagation();
	void state();
	void read_file();
	void initialize(); 

 	printf("Learning File : ");
	scanf("%s", &filename);
	//scanf("%s", filename, 20*sizeof(char));
	printf("\n 입력된 문자열: %s \n", filename);

	read_file(filename);
	initialize(); 

	printf("\n\n *****************************************\n");
	printf("\tPattern Output1 Output2 Output3 Output4\n");

 	for (errorfunc = 0.0, i = 0; i < learning_pattern_no; i++) {
		state(i);
		for (j = 0; j < OutputUnitNo; j++) {
			errorfunc += pow(t[i][j] - O3[j], 2.0);	
			
		}
	} 
	// �н�����
	printf("\n************************************************");
	printf("\nCount Pattern Output1 Output2 Output3 Output4\n");
 	// for (i = 0; errorfunc > ErrorFunc;) {
	for (i = 0; i<2;i++) {
		for (j = 0; j < learning_pattern_no; j++) {
			propagation(j);
			back_propagation(j);
		}
		for (errorfunc = 0.0, j = 0; j < learning_pattern_no; j++) {
			printf("%d", ++i);
			state(j);
			for (k = 0; k < OutputUnitNo; k++) {
				errorfunc += pow(t[j][k] - O3[k], 2.0);
			}
		}
		errorfunc /= 2;
		printf("ErrorFunc : %3f\n", errorfunc);
	}  
	printf("\n**********************�ű� ������ ���**************************\n");
	printf("\tPattern  Output1 Output2 Output3 Output4\n");
  	for(i = 0; i < (learning_pattern_no + test_pattern_no); i++){
		state(i);
	}  
}

void propagation(int p){
	 int i, j;
	double net;

	for (i = 0; i < HiddenUnitNo; i++) {
		for (net = 0.0, j = 0; j < InputUnitNo; j++) {
			net += W21[i][j] * O1[p][j];
		}
		O2[i] = f(net + bias2[i]);
	}

	for ( i = 0; i < OutputUnitNo; i++) {
		for (net = 0.0, j = 0; j < HiddenUnitNo; j++) {
			net += W32[i][j] * O2[j];
		}
		O3[i] = f(net + bias3[i]); 
	} 
}

void back_propagation(int p)
{
	int i, j;
	double d2[HiddenUnitNo];
	double d3[OutputUnitNo];
	double sum;

	for (i = 0; i < OutputUnitNo; i++) {
		d3[i] = (t[p][i] - O3[i]) * O3[i] * (1 - O3[i]);
	}
	for (i = 0; i < HiddenUnitNo; i++) {
		for (sum = 0.0, j = 0; j < OutputUnitNo; j++) {
			dW32[j][i] = Eta * d3[j] * O2[i] + Alpha * dW32[j][i];
			W32[j][i] += dW32[j][i];
			sum += d3[j] * W32[j][i];
		}
		O2[i] = O2[i] * (1 - O2[i]) * sum;
	}
	for (i = 0; i < OutputUnitNo; i++) {
		dbias3[i] = Eta * d3[i] + Alpha * dbias3[i];
		bias3[i] += dbias3[i];
	}

	for (i = 0; i < InputUnitNo; i++) {
		for (j = 0; j < HiddenUnitNo; j++) {
			dW21[j][i] = Eta * d2[j] * O1[p][i] + Alpha * dW21[j][i];
			W21[j][i] += dW21[j][i];
		}
	}

	for (i = 0; i < HiddenUnitNo; i++) {
		dbias2[i] = Eta * d2[i] + Alpha * dbias2[i];
		bias2[i] += dbias2[i];
	}
}

void state(int p) 
{
	int i;
	printf("\t%d -> ", p + 1);
	propagation(p);
	for (i = 0; i < OutputUnitNo; i++) {
		printf("\t%5.3f", O3[i]);
	}
	fputs("\n", stdout);
}

void read_file(char *name) 
{
	int i, j;
	FILE *fp;

	if ((fp = fopen(name, "r")) == NULL) {
		fprintf(stderr, "\n%s : File Open Error !!\n", name);
		exit(-1);
	}

	fscanf(fp, "%d", &learning_pattern_no);
	printf("learning_pattern_no : %d\n", learning_pattern_no);
	for (i = 0; i < learning_pattern_no; i++) {
		for (j = 0; j < learning_pattern_no; j++) {
			fscanf(fp, "%lf", &O1[i][j]);
			printf("&01[%d][%d] :  %lf\n",i, j, O1[i][j]);
		}
		for (j = 0; j < OutputUnitNo; j++) {
			fscanf(fp, "%lf", &t[i][j]);
		}
	}
	fscanf(fp, "%d", &test_pattern_no);
	printf("test_pattern_no : %d\n", test_pattern_no);
	for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++) {
		for (j = 0; j < InputUnitNo; j++){
			fscanf(fp, "%lf",&O1[i][j]);
		}
	} 
	fclose(fp);
}

void initialize() 
{
	int i, j;
	for (i = 0; i < HiddenUnitNo; i++) {
		for (j = 0; j < InputUnitNo; j++) {
			W21[i][j] = rnd();
		}
	}
	for (i = 0; i < OutputUnitNo; i++) {
		for (j = 0; j < HiddenUnitNo; j++) {
			W32[i][j] = rnd();
		}
	}
}