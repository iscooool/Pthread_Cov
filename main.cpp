#include<iostream>
#include "mpi.h"
#include<fcntl.h>
#include<sys/mman.h>
#include<unistd.h>
#include<sys/stat.h>
#include<sys/types.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <immintrin.h>
#include<string.h>
#include <stdlib.h>
//-----------------------
#define kernal_size 5
#define board_width 2
int image_hight;
int image_width;
int start_rgb;//数据开始位置
int start_line;//本进程开始行
int bytesPerLine;//每一行的字节数
//-----------------------
const float kernal[kernal_size * kernal_size] = {
        0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,
        0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,
        0.04,0.04,0.04,0.04,0.04
};
//-----------------------
using namespace std;

unsigned int get_unsignedint(char *ptr, int pos)//获取某个位置开始的无符号int 高地址高位
{
    return ((int) (unsigned char) ptr[pos]) | (((int) (unsigned char) ptr[pos + 1]) << 8) |
           (((int) (unsigned char) ptr[pos + 2]) << 16) | (((int) (unsigned char) ptr[pos + 3]) << 24);
}

void get_RGB(char *ptr, float *R, float *G, float *B, int comm_sz, int my_rank, int my_row_num, int my_col_num) {
    //根据对应的进程号分配任务
    for (int i = 0; i <= my_row_num - 1; i++)//第i行第j个像素点
        for (int j = board_width; j <= my_col_num - 1 - board_width; j++) {
            int i_inbmp = start_line + i;//在图像中的行数
            int j_inbmp = j - board_width;//在图像中的列数
            if (!(i_inbmp < 0 || i_inbmp > (image_hight - 1))) {
                B[i * my_col_num + j] = float(
                        0x000000ff & ptr[start_rgb + (image_hight - 1 - i_inbmp) * bytesPerLine + j_inbmp * 3]);
                G[i * my_col_num + j] = float(
                        0x000000ff & ptr[start_rgb + (image_hight - 1 - i_inbmp) * bytesPerLine + j_inbmp * 3 + 1]);
                R[i * my_col_num + j] = float(
                        0x000000ff & ptr[start_rgb + (image_hight - 1 - i_inbmp) * bytesPerLine + j_inbmp * 3 + 2]);
            }
        }
}


void convoulution(float *R, float *G, float *B, int ans_row_num, int ans_col_num, int my_row_num,
                  int my_col_num, float *ans_R, float *ans_G, float *ans_B,int my_rank) {
    //载入卷积核 行主序
    __m256 core_07 = _mm256_load_ps(kernal);
    __m256 core_815 = _mm256_load_ps(kernal + 8);
    __m256 core_1623 = _mm256_load_ps(kernal + 16);
    float core24 = kernal[24];
    for (int i = 0; i <= ans_row_num - 1; i++)//算第i行个答案
        for (int j = 0; j <= ans_col_num - 1; j++) {
            //R G B循环
            for (int k = 0; k <= 2; k++) {
                float *RGB_cur;//指向数据RGB数组
                if (k == 0)
                    RGB_cur = R + (i + board_width) * my_col_num + j + board_width;
                else if (k == 1)
                    RGB_cur = G + (i + board_width) * my_col_num + j + board_width;
                else
                    RGB_cur = B + (i + board_width) * my_col_num + j + board_width;


                __m256i RGB_index_07 = _mm256_set_epi32(-my_col_num, -my_col_num - 1, -my_col_num - 2,
                                                        -my_col_num * 2 + 2, -my_col_num * 2 + 1, -my_col_num * 2,
                                                        -my_col_num * 2 - 1, -my_col_num * 2 - 2);
                __m256 RGB_data_07 = _mm256_i32gather_ps(RGB_cur, RGB_index_07, 4);

                __m256i RGB_index_815 = _mm256_set_epi32(my_col_num - 2, 2, 1, 0, -1, -2, -my_col_num + 2,
                                                         -my_col_num + 1);
                __m256 RGB_data_815 = _mm256_i32gather_ps(RGB_cur, RGB_index_815, 4);

                __m256i RGB_index_1623 = _mm256_set_epi32(2 * my_col_num + 1, 2 * my_col_num, 2 * my_col_num - 1,
                                                          2 * my_col_num - 2, my_col_num + 2, my_col_num + 1,
                                                          my_col_num,
                                                          my_col_num - 1);
                __m256 RGB_data_1623 = _mm256_i32gather_ps(RGB_cur, RGB_index_1623, 4);

                //测试


                float RGB_data24 = RGB_cur[2 * my_col_num + 2];
                //相乘
                __m256 RGB_ans_07 = _mm256_mul_ps(core_07, RGB_data_07);
                __m256 RGB_ans_815 = _mm256_mul_ps(core_815, RGB_data_815);
                __m256 RGB_ans_1623 = _mm256_mul_ps(core_1623, RGB_data_1623);
                float RGB_ans_24 = core24 * RGB_data24;
                //求和
                __m256 RGB_ans = _mm256_add_ps(RGB_ans_07, RGB_ans_815);
                RGB_ans = _mm256_add_ps(RGB_ans, RGB_ans_1623);
                float final_ans_ele = 0;//元素最终的答案
                float *p = (float *) (&RGB_ans);
                for (int t = 0; t <= 7; t++)
                    final_ans_ele += p[t];
                final_ans_ele += RGB_ans_24;//!!

                if (k == 0)//写入
                    ans_R[i * ans_col_num + j] = final_ans_ele;
                else if (k == 1)
                    ans_G[i * ans_col_num + j] = final_ans_ele;
                else
                    ans_B[i * ans_col_num + j] = final_ans_ele;
            }
        }
}


int main() {

    int comm_sz, my_rank;//进程数量
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

/*    double local_start, local_finish, local_elapsed, elapsed;
    MPI_Barrier(MPI_COMM_WORLD);//计时
    local_start = MPI_Wtime();
*/
    /*以下是要被计时的部分*/
    bool bmp_status = true;//读取图片是否成功
    int fd;//mmap参数
    struct stat sb;//文件属性
    char *ptr;

    if ((fd = open("timg.bmp", O_RDWR)) < 0)//打开文件失败
        bmp_status = false;
    else if ((fstat(fd, &sb)) == -1)//获取文件属性
        bmp_status = false;
    else if ((ptr = (char *) mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)) == (char *) (-1))
        bmp_status = false;

    if (bmp_status == false) {
        MPI_Finalize();//结束
        cout << "Fail";
        exit(-1);//mmap失败
    }
    //获取图片信息
    image_hight = get_unsignedint(ptr, 22);
    image_width = get_unsignedint(ptr, 18);
    start_rgb = get_unsignedint(ptr, 10);//数据开始位置
    start_line = my_rank * (image_hight / comm_sz) - board_width;//本次任务开始的行
    bytesPerLine = ((24 * image_width + 31) >> 5) << 2;//每一行的字节数

    //进行行划分
    int my_row_num = image_hight / comm_sz + 2 * board_width;
    if (my_rank == comm_sz - 1)
        my_row_num += image_hight % comm_sz;//任务数不被整除
    int my_col_num = image_width + 2 * board_width;

    /*下面开始进行卷积运算*/

    float *R = new float[my_row_num * my_col_num]{0};
    float *G = new float[my_row_num * my_col_num]{0};
    float *B = new float[my_row_num * my_col_num]{0};


    get_RGB(ptr, R, G, B, comm_sz, my_rank, my_row_num, my_col_num);//获取RGB

    int ans_row_num = my_row_num - 2 * board_width;
    int ans_col_num = my_col_num - 2 * board_width;
    float *ans_R = new float[ans_row_num * ans_col_num]{0};//计算答案
    float *ans_G = new float[ans_row_num * ans_col_num]{0};
    float *ans_B = new float[ans_row_num * ans_col_num]{0};

    convoulution(R, G, B, ans_row_num, ans_col_num, my_row_num, my_col_num, ans_R, ans_G, ans_B,my_rank);




    /*写入文件*/
    char *ptr_out;
    char suc;
    int fd_out;
    fd_out = open("test2.bmp", O_RDWR | O_CREAT, 0666);//创建新文件
    if (my_rank == 0) {
        ftruncate(fd_out, sb.st_size);
        if ((ptr_out = (char *) mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0)) ==
            (char *) (-1)) {
            suc = 0;//mmap失败
            exit(-1);
        } else
            suc = 1;//mmap成功

        for (int i = 1; i < comm_sz; i++)
            MPI_Send(&suc, 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);

        memcpy(ptr_out, ptr, start_rgb);//拷贝bmp头

    } else {
        MPI_Recv(&suc, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (suc == 0)
            exit(-1);
        ptr_out = (char *) mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);
    }

    for (int i = 0; i <= ans_row_num - 1; i++) {
        for (int j = 0; j <= ans_col_num - 1; j++) {
            int pos_pix = start_rgb + (image_hight - 1 - (my_rank * (image_hight / comm_sz) + i))
                                      * bytesPerLine + j * 3;
            ptr_out[pos_pix] = (char) (ans_B[i * ans_col_num + j]);
            ptr_out[pos_pix + 1] = (char) (ans_G[i * ans_col_num + j]);
            ptr_out[pos_pix + 2] = (char) (ans_R[i * ans_col_num + j]);
        }
    }
 /*   local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
        cout << "time:" << elapsed << "s" << endl;*/
    MPI_Finalize();//结束
    return 0;
}

