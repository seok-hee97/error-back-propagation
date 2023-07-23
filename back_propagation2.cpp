#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma warning(disable : 4996)

#define InputUnitNo 2   // 입력층 노드 수
#define HiddenUnitNo 3  // 은익층 노드 수
#define OutputUnitNo 1  // 출력층 노드 수
#define MaxPatternNo 4  // 학습 패턴 최대수

#define Eta 0.75
#define Alpha 0.25
#define ErrorFunc 0.05
#define Wmin -0.30
#define Wmax 0.30

#define f(x) (1 / (1 + exp(-(x))))
#define rnd() ((double)rand() / 0x7fff * (Wmax - Wmin) + Wmin)

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

void propagation(int p);
void back_propagation(int p);
void state(int p);
void read_file(char* name);
void initialize();

void main() {
  int i, j, k;
  char filename[30];
  char ss[80];
  double errorfunc;

  printf("Learning File : ");
  scanf("%s", &filename);
  printf("\n # 입력한 파일명: %s", filename);

  read_file(filename);
  initialize();

  printf(
      "\n\n *****************************학습하기 "
      "전*****************************\n");
  printf("\tPattern Output1 Output2 Output3 Output4\n");

  for (errorfunc = 0.0, i = 0; i < learning_pattern_no; i++) {
    state(i);
    for (j = 0; j < OutputUnitNo; j++)
      errorfunc += pow(t[i][j] - O3[j], 2.0);
  }

  errorfunc /= 2;
  printf(" ErrorFunc : %.3f\n", errorfunc);

  printf(
      "\n*****************************학습 시작 "
      "*****************************\n");
  printf("Count Pattern Output1 Output2 Output3 Output4\n");
  for (i = 0; errorfunc > ErrorFunc;) {
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
    printf("ErrorFunc : %.3f\n", errorfunc);
  }

  printf(
      "\n***************************** 신규 데이터 결과 "
      "*****************************\n");
  printf("\tPattern Output1 Output2 Output3 Output4\n");
  for (i = 0; i < learning_pattern_no + test_pattern_no; i++)
    state(i);

  system("PAUSE");
}

void propagation(int p) {
  int i, j;
  double net;
  /* 은익층 노드에서 출력 계산 */
  for (i = 0; i < HiddenUnitNo; i++) {
    for (net = 0.0, j = 0; j < InputUnitNo; j++)
      net += W21[i][j] * O1[p][j];

    O2[i] = f(net + bias2[i]);
  }
  /* 출력 노드에서 출력 계산 */
  for (size_t i = 0; i < OutputUnitNo; i++) {
    for (net = 0.0, j = 0; j < HiddenUnitNo; j++)
      net += W32[i][j] * O2[j];

    O3[i] = f(net + bias3[i]);
  }
}

void back_propagation(int p) {
  int i, j;
  double d2[HiddenUnitNo];
  double d3[OutputUnitNo];
  double sum;

  for (i = 0; i < OutputUnitNo; i++)
    d3[i] = (t[p][i] - O3[i]) * O3[i] * (1 - O3[i]);

  for (i = 0; i < HiddenUnitNo; i++) {
    for (sum = 0.0, j = 0; j < OutputUnitNo; j++) {
      dW32[j][i] = Eta * d3[j] * O2[i] + Alpha * dW32[j][i];
      W32[j][i] += dW32[j][i];
      sum += d3[j] * W32[j][i];
    }
    d2[i] = O2[i] * (1 - O2[i]) * sum;
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

void state(int p) {
  int i;
  printf("\t%d -> ", p + 1);
  propagation(p);
  for (i = 0; i < OutputUnitNo; i++)
    printf("\t%5.3f", O3[i]);

  fputs("\n", stdout);
}

void read_file(char* name) {
  int i, j;
  FILE* fp;
  if ((fp = fopen(name, "r")) == NULL) {
    fprintf(stderr, "\n%s : File Open Error!!\n", name);
    exit(-1);
  }
  fscanf(fp, "%d", &learning_pattern_no);

  for (i = 0; i < learning_pattern_no; i++) {
    for (j = 0; j < InputUnitNo; j++)
      fscanf(fp, "%lf", &O1[i][j]);

    for (j = 0; j < OutputUnitNo; j++)
      fscanf(fp, "%lf", &t[i][j]);
  }

  fscanf(fp, "%d", &test_pattern_no);

  for (i = learning_pattern_no; i < learning_pattern_no + test_pattern_no; i++) {
    for (j = 0; j < InputUnitNo; j++)
      fscanf(fp, "%lf", &O1[i][j]);
  }

  fclose(fp);
}

void initialize() {
  int i, j;
  for (i = 0; i < HiddenUnitNo; i++) {
    for (j = 0; j < InputUnitNo; j++)
      W21[i][j] = rnd();
  }

  for (i = 0; i < OutputUnitNo; i++) {
    for (j = 0; j < HiddenUnitNo; j++)
      W32[i][j] = rnd();
  }
}
