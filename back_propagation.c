#define _CRT_SECURE_NO_WARNINGS


#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>


/* 네트워크의 파라메터 */
#define InputUnitNo      2
#define HiddenUnitNo   3
#define OutputUnitNo   1
#define MaxPatternNo   4

#define Eta            0.75
#define Alpha         0.25
#define ErrorFunc      0.05
#define Wmin         -0.30
#define Wmax         0.30

/* Sigmoid & Random Numbers */
#define f(x) (1/(1+exp(-(x))))
#define rnd() ((double)rand()/0x7fff * (Wmax-(Wmin))+(Wmin))

/* 네트워크 구성 */
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
int lerning_pattern_no;
int test_pattern_no;

/* 메인 프로그램 */
//void main(int argc, char *argv[])
int main()
{
   int i, j, k;
   char filename[30];
   char ss[80];
   double errorfunc;

   /* 긴 프로그램을 단순화 하기 위한 함수로 분리 */
   void propagation();
   void back_propagation();
   void state();
   void read_file();
   void initialize();

   /* 데이터 파일명 읽기 문 */
   printf("Learning File : ");
   scanf("%s", &filename);
   printf("\n # 입력한 파일명: %s", filename);

   /* 파일명을 받아서 데이터 정리 */
   read_file(filename);
   initialize();

   printf("\n\n *************** 학습하기 전 ***************\n");
   printf("\tPattern Ouput1 Output2 Output3 Output4\n");

   /* 학습 패턴에 대한 학습 및 결과값 표시 */
   for (errorfunc = 0.0, i = 0; i < lerning_pattern_no; i++)
   {
      state(i);
      for (j = 0; j < OutputUnitNo; j++)
      {
         errorfunc += pow(t[i][j] - O3[j], 2.0);
      }
   }
   errorfunc /= 2;
   printf(" ErrorFunc : %.3f\n", errorfunc);

   /* 학습 시작 */
   printf("\n*************** 학습시작 ***************");
   printf("\nCount Pattern Output1   Output2   Output3   Output4\n");
   for (i = 0; errorfunc > ErrorFunc;)
   {
      for (j = 0; j < lerning_pattern_no; j++)
      {
         propagation(j);
         back_propagation(j);
      }
      for (errorfunc = 0.0, j = 0; j < lerning_pattern_no; j++)
      {
         printf("%d", ++i);
         state(j);
         for (k = 0; k < OutputUnitNo; k++)
         {
            errorfunc += pow(t[j][k] - O3[k], 2.0);
         }
      }
      errorfunc /= 2;
      printf("ErrorFunc : %3.f\n", errorfunc);
   }
   /* 학습 종료 */
   printf("\n *************** 신규 데이터 결과 ***************\n");
   printf("\tPattern Ouput1   Output2   Output3   Output4\n");

   /* 신규 데이터 예측 결과 */
   for (i = 0; i < lerning_pattern_no + test_pattern_no; i++)
   {
      state(i);
   }
}

/* 순방향 입력층에서 출력층으로 */
void propagation(p)
int p;
{
   int i, j;
   double net;

   /* 은익층 노드에서 출력 계산 */
   for (i = 0; i < HiddenUnitNo; i++)
   {
      for (net = 0.0, j = 0; j < InputUnitNo; j++)
      {
         net += W21[i][j] * O1[p][j];
      }
      O2[i] = f(net + bias2[i]);
   }

   /* 출력 노드에서 출력 계산 */
   for (size_t i = 0; i < OutputUnitNo; i++)
   {
      for (net = 0.0, j = 0; j < HiddenUnitNo; j++)
      {
         net += W32[i][j] * O2[j];
      }
      O3[i] = f(net + bias3[i]);
   }
}

/* 역방향 가중치 변경을 위한 출력층에서 입력층으로 */
void back_propagation(p)
int p;
{
   int i, j;
   double d2[HiddenUnitNo];
   double d3[OutputUnitNo];
   double sum;

   /* 평균제곱오차: dj=(oj-yj) f'(i,j) */
   for (i = 0; i < OutputUnitNo; i++)
   {
      d3[i] = (t[p][i] - O3[i]) * O3[i] * (1 - O3[i]);
   }

   /* 가중치 변경 w_(i,j) = nd_(j) O_(i) */
   for (i = 0; i < HiddenUnitNo; i++)
   {
      for (sum = 0.0, j = 0; j < OutputUnitNo; j++)
      {
         dW32[j][i] = Eta * d3[j] * O2[i] + Alpha * dW32[j][i];
         W32[j][i] += dW32[j][i];
         sum += d3[j] * W32[j][i];
      }
      /* d_(j) = (∑ w_(j,i) d_(i)) f'(i,j) */
      d2[i] = O2[i] * (1 - O2[i]) * sum;
   }
   for (i = 0; i < OutputUnitNo; i++)
   {
      dbias3[i] = Eta * d3[i] + Alpha * dbias3[i];
      bias3[i] += dbias3[i];
   }
   /* w_(i,j) = nd_(j) O_(i) */
   for (i = 0; i < InputUnitNo; i++)
   {
      for (j = 0; j < HiddenUnitNo; j++)
      {
         dW21[j][i] = Eta * d2[j] * O1[p][i] + Alpha * dW21[j][i];
         W21[j][i] += dW21[j][i];
      }
   }
   for (i = 0; i < HiddenUnitNo; i++)
   {
      dbias2[i] = Eta * d2[i] + Alpha * dbias2[i];
      bias2[i] += dbias2[i];
   }
}

/* 결과 상태 표시 */
void state(int p)
{
   int i;
   printf("\t%d -> ", p + 1);
   propagation(p);
   for (i = 0; i < OutputUnitNo; i++)
   {
      printf("\t%5.3f", O3[i]);
   }
   fputs("\n", stdout);
}

/*데이터 파일 읽기*/
void read_file(char *name) {
   int i, j;
   FILE* fp;
   /*파일 열기*/
   if ((fp = fopen(name, "r")) == NULL) {
      fprintf(stderr, "\n%s : File Open Error!!\n", name);
      exit(-1);
   }
   /*학습 데이터 읽어 오기*/
   fscanf(fp, "%d", &lerning_pattern_no);
   for (i = 0; i < lerning_pattern_no; i++)
   {
      for (j = 0; j < InputUnitNo; j++) {
         fscanf(fp, "%lf", &O1[i][j]);
      }
      for (j = InputUnitNo; j< +  OutputUnitNo; j++) {
         fscanf(fp, "%lf", &t[i][j]);
      }
   }
   /*테스트 데이터 읽어 오기*/
   fscanf(fp, "%d", &test_pattern_no);
   for (i = lerning_pattern_no; i < lerning_pattern_no + test_pattern_no; i++)
   {
      for (j = 0; j < InputUnitNo; j++)
      {
         fscanf(fp, "%lf", &O1[i][j]);
      }
   }
   fclose(fp);

}
/* 가중치 초기화 */
void initialize() {
   int i, j;
   /* 가중치 초기화 입력층 -> 은익층*/
   for (i = 0; i < HiddenUnitNo; i++) {
      for (j = 0; j < InputUnitNo; j++) {
         W21[i][j] = rnd();
      }
   }
   /*가중치 초기화 은익층 -> 출력층 */
   for (i = 0; i < OutputUnitNo; i++)
   {
      for (j = 0; j < HiddenUnitNo; j++) {
         W32[i][j] = rnd();
      }
   }
}