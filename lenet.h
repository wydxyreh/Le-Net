//my lenet
/*
输入为1*10
卷积层C1，12个3*1的卷积核。输出为12个1*8的向量
池化层S1，平均池化区域为1*2。输出为12个1*4的向量
全连接层fc1，48->24
全连接层fc2，24->10
*/
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include "data_input.h"
using namespace std;
ofstream dataFile;
ifstream weights_in;
ofstream weights_out;

//计算e的x次方
double exp(double x)
{
 	x = 1.0 + x/256;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

float expf(float x) {
	x = 1.0 + x / 1024;
	x *= x; x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x; x *= x;
	return x;
}

//平均池化
// 输入2个数据，2个数据相加，求平均值得到一个结果。
float AvgPool_1x2(float input[2]){
	float res = 0;
	int i;
	for(i = 0; i < 2 ; i++){
		res += input[i];
	}
	res /= 2;
	return res;
}

//sigmoid函数实现
float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

//激励函数实现
float inspirit(float x, float y, float A, float B){
    return (x-y)*x*(A-x)/(A*B);
}

//3*1卷积核 卷积运算
float Conv_3x1(float input[3], float kernel[3]){
	int x;
	float result = 0;
	for(x = 0; x < 3; x++){
		result += input[x] * kernel[x];
	}
	return result;
}

//卷积层C1
//输入为1*10的矩阵，输入数据类型为浮点型
//kernel 3x1x12 = 3x12 = 36
void ConvLayer_1(float input[10],float * C1_value,float * weights){
	int i_x,matrix_x;  // 循环变量和中间变量
	int k_num,mat_i = 0;            // 不同kernel的计数变量

	// 定义循环名称为top_loop，方便优化
	// k_num为不同卷积核的循环变量，第一层有12个不同的卷积核，自然循环12次产生12张feature map
	top_loop:for(int k_num = 0; k_num < 12; k_num+=1){
		// 卷积核的权重数据变量matrix_2，用来存放kernel数据的
		// 第一层C1有12个不同的kernel，利用外层的top_loop可以完成对这些不同的kernel进行赋值
		// 3x1的kernel有3个变量，循环3次来放
		float matrix_2[3];
		for(mat_i = 0;mat_i<3;mat_i++){
			matrix_2[mat_i] = weights[mat_i + k_num*3];//weight是长度为3*12的向量
		}
		// 一次卷积核操作，完成1*8的输出
		i_x_loop:for(i_x = 0; i_x < 8; i_x++){
            float matrix[3];
            int value_index = i_x;
            // 通过25次循环，先把一个点乘运算所需的输入图像数据给弄出来，放到matrix中
            matrix_loop:for(matrix_x = 0; matrix_x <3; matrix_x++){
                // 索引是0 ~ 2，这个也好理解，有3个数据
                int matrix_index = matrix_x;
                // 图片像素索引 0 ~ 1024。与matrix_x，matrix_y相关，x、y=32。
                int input_value_index = value_index + matrix_x;
                matrix[matrix_index] = input[input_value_index];
            }
            // out_index为输出数据的索引，从之前的学习，可以知道
            // C1后将输出1x8x12=96个数据，而每个数据，可以由out_index索引
            int out_index = i_x + k_num * 8;
            // 通过最基本的卷积点乘单元，输入为图像数据和卷积核值。
            C1_value[out_index] = sigmoid(Conv_3x1(matrix,matrix_2));
		}
	}
}

//池化层S1
//池化层输入为1*8*12，输出保存到A2_value
void AvgpoolLayer_2(float input[96],float *A2_value){
	int k_num,i_x,matrix_x;
	int count = 0;
	// 有12张feature map，需要循环12次
	for(k_num = 0; k_num < 12; k_num++){
		// 一张feature map大小为1x8，需要把每个数据给遍历了
		for(i_x = 0; i_x < 8; i_x+=2){
            float matrix[2];
            // 此时1x8x12某个图像数据的索引
            int index_now = i_x + k_num * 8;
            // 1x2的区域内，做一个平均池化
            for(matrix_x = 0; matrix_x < 2; matrix_x++){
                // 将输出的索引转化为输入图像数据的索引，类似一个值映射变为2个值
                // 把2个索引里的值放到之前定义的matrix变量中，用于计算。
                int input_index = index_now + matrix_x;
                matrix[matrix_x] = input[input_index];
            }
            // 将1x8个数据遍历完，计算2个产生的一个结果，具体是2个数据相加
            // 加后乘上一个1/2，为均值，同时偏置设为了0，最后将结果通入sigmoid函数
            // 为了方便，这里并没有对池化中相加后乘的那个均值数做训练，且偏置也没有训练
            A2_value[count] = sigmoid(AvgPool_1x2(matrix));
            count++;  // 计数变量增加，来完成1x8遍历下的所有输出
		}
	}
}

//两层全连接层
//kernel 48x24 = 1152
void FullyConnLayer_3(float input[48],float *F3_value,float * weights){
	int i_x,i_y;
	for(i_x = 0; i_x < 24; i_x++){
		float res = 0;
		for(i_y = 0;  i_y < 48; i_y++){
			int index = i_y + i_x * 48;
			res += input[i_y] * weights[index + 48];
		}
		F3_value[i_x] = sigmoid(res+weights[i_x+1200]);
	}
}
//kernel 24x10 = 240
void FullyConnLayer_4(float input[24],float *F4_value,float * weights){
	int i_x,i_y;
	for(i_x = 0; i_x < 10; i_x++){
		float res = 0;
		for(i_y = 0;  i_y < 24; i_y++){
			int index = i_y + i_x * 24;
			res += input[i_y] * weights[index + 1248];
		}
		F4_value[i_x] = sigmoid(res+weights[i_x+1488]);
	}
}

//softmax函数实现
/*
int Softmax_1_8(float input[10],float *probability,float *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index;
}
*/

//photo是标准输出，data是权重，output是预测输出
void feedback(float std[10], float *data, float output[10], float A2_value[48], float F3_value[24], float ETA_W, float ETA_B, float A, float B){
	//layer1 weights 3x1x12 = 3x12 = 36
	//layer1 bias1	 1*12   = 12
	//layer3 weights 48x24  = 1152
	//layer3 bias2 	 1x48   = 48
	//layer4 weights 24x10  = 240
	//layer3 bias3	 1x24   = 24

	/*
	bias为1*1的矩阵，有12个
	ly1为3*1的矩阵，有12个
	ly3为48*24的矩阵，只有1个
	ly4为24*10的矩阵，只有1个

	ly1输出为C1=1*8矩阵，有12个
	ly3输出为F3=1*24矩阵，只有1个
	ly4输出为F4=1*10矩阵，只有1个
	
	误差函数=1/2*(y-output)^2

	error3=y-F4=1*10的矩阵
	delta3=error3*F4=1*10的矩阵
	error2=delta3.*ly4·T=1-10*10-24=1-24
	delta2=error2*F3=1*24的矩阵
	error1=delta2.*ly3·T=1-24*24=48=1-48
	delta1=(error1/12*2)*C1=12个1-8的矩阵
	
	ly4+=F3·T.*delta3=24-1*1-10=24-10
	ly3+=F2·T.*delta2=48-1*1-24=48-24

	ly1+=input·T.*delta1=10-1*1-8=10-8
	*/

	//反向传播过程
	float output_delta[10];
	float ly4_delta[24];
	float ly3_delta[48];
	float ly1_delta[12][3];
	int i_x,i_y;
	float res=0;

	//输出层误差，output_delta为1*10矩阵
	for(int i=0;i<10;i++)
		output_delta[i]=inspirit(output[i],std[i],A,B);
	//全连接层2误差，ly4_delta为1*24矩阵
	for(i_x=0;i_x<24;i_x++){
		res=0;
		for(i_y=0;i_y<10;i_y++){
			res+=data[i_x*10+i_y+1248]*output_delta[i_y];
		}
		ly4_delta[i_x]=res*F3_value[i_x]*(A-F3_value[i_x])/(A*B);
	}
	//全连接层1误差，ly3_delta为1*48矩阵
	for(i_x=0;i_x<48;i_x++){
		res=0;
		for(i_y=0;i_y<24;i_y++){
			res+=data[i_x*24+i_y+48]*ly4_delta[i_y];
		}
		ly3_delta[i_x]=res*A2_value[i_x]*(A-A2_value[i_x])/(A*B);
	}
	//池化层（平均池化）反向传播，针对1*48的向量进行分割，获得卷积层的输出矩阵，即12个1*8的矩阵
	int delta_conv[12][8];
	for(int i=0;i<12;i++){
		delta_conv[i][0]=ly3_delta[i*4+0];
		delta_conv[i][1]=ly3_delta[i*4+0];
		delta_conv[i][2]=ly3_delta[i*4+1];
		delta_conv[i][3]=ly3_delta[i*4+1];
		delta_conv[i][4]=ly3_delta[i*4+2];
		delta_conv[i][5]=ly3_delta[i*4+2];
		delta_conv[i][6]=ly3_delta[i*4+3];
		delta_conv[i][7]=ly3_delta[i*4+3];
	}
	//卷积层误差，将权重看作输入，输入1*10的向量以输入转化为3*8的矩阵
	int fc_in[3][8];
	for(int i=0;i<8;i++){
		for(int j=0;j<3;j++){
			fc_in[j][i]=data[i+j+1512];
		}
	}
	//此时w-in-out构成了一个全连接层，w为1*3的矩阵输入，out为1*8的矩阵输出，in为3*8的权重向量
	for(int i=0;i<12;i++){
		for(i_x=0;i_x<3;i_x++){
			res=0;
			for(i_y=0;i_y<8;i_y++){
				res+=fc_in[i_x][i_y]*delta_conv[i][i_y];
			}
			ly1_delta[i][i_x]=res*data[i_x+i*3]*(A-data[i_x+i*3])/(A*B);
		}
	}

	//求解权重变化值
	//卷积层
	for(int i=12;i<0;i++){
		for(int j=0;j<3;j++){
			data[i*3+j]+=ETA_W*ly1_delta[i][j];
		}
	}
	//全连接层1
	for(int i=0;i<48;i++){
		for(int j=0;j<24;j++){
			data[i*24+j]-=ETA_W*ly4_delta[j]*A2_value[i];
			data[1200+j]-=ETA_B*ly4_delta[j];
		}
	}
	//全连接层2
	for(int i=0;i<24;i++){
		for(int j=0;j<10;j++){
			data[i*10+j]-=ETA_W*output_delta[j]*F3_value[i];
			data[1488+j]-=ETA_B*output_delta[j];
		}
	}
}

//Le-Net神经网络
void LetNet(float photo[10], float *data, float *output, float *fc2, float *fc1){
/*
#pragma HLS INTERFACE m_axi depth=62855 port=addrMaster offset=slave bundle=MASTER_BUS
#pragma HLS INTERFACE s_axilite port=r bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
*/
	//1*10 input
	//float photo[10];
	//layer1 weights 3x1x12 = 3x12 = 36
	//layer1 bias1	 1*12   = 12
	//layer3 weights 48x24  = 1152
	//layer3 bias2 	 1x48   = 48
	//layer4 weights 24x10  = 240
	//layer3 bias3	 1x24   = 24
	
    //float data[1522];
	//The output of each layer
	float C1_value[96];
	float A2_value[48];
	float F3_value[24];
	//float F4_value[10];

	//float probability[10];
	//float res[10];
	int loop1_i;
	//memory copy from BRAM to FPGA's RAM, 从addrMaster地址开始，复制1522个浮点型的数据到data数组中去。
	//memcpy(data,(const float*)addrMaster,1522*sizeof(float));
	//get the image data, 1*10=10 datas
	for(loop1_i = 0; loop1_i<10; loop1_i++){
		photo[loop1_i] = data[loop1_i+1512];  // 前1512放的都是权重数据，后面才是数据
	}
	//calulation of each layer
	ConvLayer_1(photo,C1_value,data);
	AvgpoolLayer_2(C1_value,A2_value);
	FullyConnLayer_3(A2_value,F3_value,data);
	FullyConnLayer_4(F3_value,output,data);

	//*r = Softmax_1_8(F7_value,probability,res);
}

void out_to_file(float input[873][10], int num){
	switch(num){
		case 0:dataFile.open("dataFile_0.2_60.txt", ios::in | ios::out | ios::trunc);break;
		case 1:dataFile.open("dataFile_0.2_70.txt", ios::in | ios::out | ios::trunc);break;
		case 2:dataFile.open("dataFile_0.2_80.txt", ios::in | ios::out | ios::trunc);break;
		case 3:dataFile.open("dataFile_0.2_90.txt", ios::in | ios::out | ios::trunc);break;
		case 4:dataFile.open("dataFile_0.2_100.txt", ios::in | ios::out | ios::trunc);break;
		case 5:dataFile.open("dataFile_0.4_60.txt", ios::in | ios::out | ios::trunc);break;
		case 6:dataFile.open("dataFile_0.4_70.txt", ios::in | ios::out | ios::trunc);break;
		case 7:dataFile.open("dataFile_0.4_80.txt", ios::in | ios::out | ios::trunc);break;
		case 8:dataFile.open("dataFile_0.4_90.txt", ios::in | ios::out | ios::trunc);break;
		case 9:dataFile.open("dataFile_0.4_100.txt", ios::in | ios::out | ios::trunc);break;
		case 10:dataFile.open("dataFile_0.6_60.txt", ios::in | ios::out | ios::trunc);break;
		case 11:dataFile.open("dataFile_0.6_70.txt", ios::in | ios::out | ios::trunc);break;
		case 12:dataFile.open("dataFile_0.6_80.txt", ios::in | ios::out | ios::trunc);break;
		case 13:dataFile.open("dataFile_0.6_90.txt", ios::in | ios::out | ios::trunc);break;
		case 14:dataFile.open("dataFile_0.6_100.txt", ios::in | ios::out | ios::trunc);break;
		case 15:dataFile.open("dataFile_0.6_60.txt", ios::in | ios::out | ios::trunc);break;
		case 16:dataFile.open("dataFile_0.8_70.txt", ios::in | ios::out | ios::trunc);break;
		case 17:dataFile.open("dataFile_0.8_80.txt", ios::in | ios::out | ios::trunc);break;
		case 18:dataFile.open("dataFile_0.8_90.txt", ios::in | ios::out | ios::trunc);break;
		case 19:dataFile.open("dataFile_0.8_100.txt", ios::in | ios::out | ios::trunc);break;
		case 20:dataFile.open("dataFile_1.0_60.txt", ios::in | ios::out | ios::trunc);break;
		case 21:dataFile.open("dataFile_1.0_70.txt", ios::in | ios::out | ios::trunc);break;
		case 22:dataFile.open("dataFile_1.0_80.txt", ios::in | ios::out | ios::trunc);break;
		case 23:dataFile.open("dataFile_1.0_90.txt", ios::in | ios::out | ios::trunc);break;
		case 24:dataFile.open("dataFile_1.0_100.txt", ios::in | ios::out | ios::trunc);break;
		default:break;
	}
	for(int i=0;i<873;i++){
		for(int j=0;j<10;j++){
			// 朝TXT文档中写入数据
			dataFile << input[i][j] << " ";
		}
		dataFile << endl;
	}
	// 关闭文档
	dataFile.close();
}

void read_weights(float  data[1522]){
	weights_in.open("dataFile_weights.txt", ios::in);
	for(int i=0;i<1522;i++){
		weights_in >> data[i];
	}
}

void write_weights(float  data[1522]){
	weights_out.open("dataFile_weights.txt", ios::out | ios::trunc);
	for(int i=0;i<1522;i++){
		weights_out << data[i] <<" ";
	}
}


void trian_ETA_W_B(float *best_ETA_W, float *best_ETA_B, float *photo, float *data, float *output, float *fc2, float *fc1, float *std, float A, float B){
	float min_res=10,res,ans;
	float **data_input = new float*[873];
    for(int i=0;i<873;i++)
	    data_input[i] = new float[10];
	float data_output[873][10];
	//ETA_B和ETA_W从0.1开始到10.0，每次增加0.1，选择误差函数值最小的保存起来
	for(float ETA_B=0.1;ETA_B<10.00;ETA_B+=0.1){
        for(float ETA_W=0.1;ETA_W<10.00;ETA_W+=0.1){
            res=0;ans=0;
            for(int i=0;i<10;i++)
                data_output[0][i]=0;
            read_weights(data);//读取已有权重
            get_input_1_06_80(data_input);//以06_80这一组为例，共873组，每组为1*10的向量
            //train 643
            for(int i=0;i<643;i++){
                for(int loop1_i=0;loop1_i<10;loop1_i++){
                    data[loop1_i+1512]=data_input[i][loop1_i];//获取当前的输入，储存到data的最后10位
                    std[loop1_i]=data_input[i+1][loop1_i];//获取当前输入的真实输出（即下一组数据
                }
                LetNet(photo,data,output,fc1,fc2);//前向传播
                for(int j=0;j<10;j++){
                    data_output[i+1][j]=output[j];//保存预测结果
                }
                feedback(std,data,output,fc1,fc2,ETA_W,ETA_B,A,B);//反向传播
            }
            //test 230
            for(int loop1_i=0;loop1_i<10;loop1_i++){
                data[loop1_i+1512]=data_input[643][loop1_i];//获取当前的输入，储存到data的最后10位
            }
            for(int i=644;i<873;i++){
                LetNet(photo,data,output,fc1,fc2);//前向传播
                for(int j=0;j<10;j++){
                    data_output[i][j]=output[j];//保存预测结果
                    data[j+1512]=output[j];//将当前预测结果，作为下一次预测的输入
                    //计算预测误差
                    ans+=0.5*(data_input[i][j]-output[j])*(data_input[i][j]-output[j]);
                }
                res+=ans;
                ans=0;
            }
            //cout <<endl <<endl <<"res:" <<res/230 <<endl;
            if(res<min_res){
                min_res=res;//每组ETA_W和ETA_B训练完成之后，比较并存储最低的误差函数
                *best_ETA_W=ETA_W;//更新最优的ETA_W
                *best_ETA_B=ETA_B;//更新最优的ETA_B
                write_weights(data);//将当前最优权重写入文件中保存
            }
        }
    }
    cout <<min_res <<endl;
}

void trian_sigmoid_A_B(float *best_A, float *best_B, float *photo, float *data, float *output, float *fc2, float *fc1, float *std, float ETA_W, float ETA_B){
	float min_res=10,res,ans;
	float **data_input = new float*[873];
    for(int i=0;i<873;i++)
	    data_input[i] = new float[10];
	float data_output[873][10];

	//A和B从1开始到30，每次增加0.1，选择误差函数值最小的保存起来
	for(float A=1;A<30;A+=0.1){
        for(float B=1;B<30;B+=0.1){
            res=0;ans=0;
            for(int i=0;i<10;i++)
                data_output[0][i]=0;
            read_weights(data);//读取已有权重
            get_input_1_06_80(data_input);//以06_80这一组为例，共873组，每组为1*10的向量
            //train 643
            for(int i=0;i<643;i++){
                for(int loop1_i=0;loop1_i<10;loop1_i++){
                    data[loop1_i+1512]=data_input[i][loop1_i];//获取当前的输入，储存到data的最后10位
                    std[loop1_i]=data_input[i+1][loop1_i];//获取当前输入的真实输出（即下一组数据
                }
                LetNet(photo,data,output,fc1,fc2);//前向传播
                for(int j=0;j<10;j++){
                    data_output[i+1][j]=output[j];//保存预测结果
                }
                feedback(std,data,output,fc1,fc2,ETA_W,ETA_B,A,B);//反向传播
            }
            //test 230
            for(int loop1_i=0;loop1_i<10;loop1_i++){
                data[loop1_i+1512]=data_input[643][loop1_i];//获取当前的输入，储存到data的最后10位
            }
            for(int i=644;i<873;i++){
                LetNet(photo,data,output,fc1,fc2);//前向传播
                for(int j=0;j<10;j++){
                    data_output[i][j]=output[j];//保存预测结果
                    data[j+1512]=output[j];//将当前预测结果，作为下一次预测的输入
                    //计算预测误差
                    ans+=0.5*(data_input[i][j]-output[j])*(data_input[i][j]-output[j]);
                }
                res+=ans;
                ans=0;
            }
            //cout <<endl <<endl <<"res:" <<res/230 <<endl;
            if(res<min_res){
                min_res=res;//每组A和B训练完成之后，比较并存储最低的误差函数
                *best_A=A;//更新最优的A
                *best_B=B;//更新最优的B
                write_weights(data);//将当前最优权重写入文件中保存
            }
        }
    }
    cout <<min_res <<endl;
}

void get_best_weights(float *data){
	//layer1 weights 3x1x12 = 3x12 = 36
	//layer1 bias1	 1*12   = 12
	//layer3 weights 48x24  = 1152
	//layer3 bias2 	 1x48   = 48
	//layer4 weights 24x10  = 240
	//layer3 bias3	 1x24   = 24
	float my_weights[1522]={
		0.041,0.467,0.334,0.5,0.169,0.724,0.478,0.358,0.962,0.464,0.705,0.145,0.281,0.827,0.961,0.491,0.995,0.942,0.827,0.436,0.391,0.604,0.902,0.153,0.292,0.382,0.421,0.716,0.718,0.895,0.447,0.726,0.771,0.538,0.869,0.912,0.667,0.299,0.035,0.894,0.703,0.811,0.322,0.333,0.673,0.664,0.141,0.711,0.253,0.868,0.547,0.644,0.662,0.757,0.037,0.859,0.723,0.741,0.529,0.778,0.316,0.035,0.19,0.842,0.288,0.106,0.04,0.942,0.264,0.648,0.446,0.805,0.89,0.729,0.37,0.35,0.006,0.101,0.393,0.548,2.64056e+10,3.02533e+10,2.45605e+10,3.07524e+10,3.16068e+10,3.10921e+10,3.1975e+10,3.48696e+10,3.06464e+10,2.54049e+10,0.944,0.439,0.626,0.323,0.537,0.538,0.118,0.082,0.929,0.541,0.833,0.115,0.639,0.658,-4.46376e+06,0.93,0.977,0.306,0.673,0.386,0.021,0.745,0.924,0.072,0.27,0.829,0.777,0.573,0.097,0.512,0.986,0.29,0.161,0.636,0.355,0.767,0.655,0.574,0.031,0.052,0.35,0.15,0.941,0.724,0.966,0.43,0.107,0.191,0.007,0.337,0.457,0.287,0.753,0.383,0.945,0.909,0.209,0.758,0.221,0.588,0.422,0.946,-8.4921e+07,0.03,0.413,0.168,0.9,0.591,0.762,0.655,0.41,0.359,0.624,0.537,0.548,0.483,0.595,0.041,0.602,0.35,0.291,0.836,0.374,0.02,0.596,0.021,0.348,0.199,0.668,0.484,0.281,0.734,0.053,0.999,0.418,0.938,0.9,0.788,0.127,0.467,0.728,0.893,0.648,0.483,0.807,0.421,0.31,0.617,0.813,0.514,-5.93907e+06,0.616,0.935,0.451,0.6,0.249,0.519,0.556,0.798,0.303,0.224,0.008,0.844,0.609,0.989,0.702,0.195,0.485,0.093,0.343,0.523,0.587,0.314,0.503,0.448,0.2,0.458,0.618,0.58,0.796,0.798,0.281,0.589,0.798,0.009,0.157,0.472,0.622,0.538,0.292,0.038,0.179,0.19,0.657,0.958,0.191,0.815,0.888,0.156,0.511,0.202,0.634,0.272,0.055,0.328,0.646,0.362,0.886,0.875,0.433,0.869,0.142,0.844,0.416,0.881,0.998,0.322,0.651,0.021,0.699,0.557,0.476,0.892,0.389,0.075,0.712,0.6,0.51,0.003,0.869,0.861,0.688,0.401,0.789,0.255,0.423,0.002,0.585,0.182,0.285,0.088,0.426,0.617,0.757,0.832,0.932,0.169,0.154,0.721,0.189,0.976,0.329,0.368,0.692,0.425,0.555,0.434,0.549,0.441,0.512,0.145,0.06,0.718,0.753,0.139,0.423,0.279,0.996,0.687,0.529,0.549,0.437,0.866,0.949,0.193,0.195,0.297,0.416,0.286,0.105,0.488,0.282,0.455,0.734,0.114,0.701,0.316,0.671,0.786,0.263,0.313,0.355,0.185,0.053,0.912,0.808,0.832,0.945,0.313,0.756,0.321,0.558,0.646,0.982,0.481,0.144,0.196,0.222,0.129,0.161,0.535,0.45,0.173,0.466,0.044,0.659,0.292,0.439,0.253,0.024,0.154,0.51,0.745,0.649,0.186,0.313,0.474,0.022,0.168,0.018,0.787,0.905,0.958,0.391,0.202,0.625,0.477,0.414,0.314,0.824,0.334,0.874,0.372,0.159,0.833,0.07,0.487,0.297,0.518,0.177,0.773,0.27,0.763,0.668,0.192,0.985,0.102,0.48,0.213,0.627,0.802,0.099,0.527,0.625,0.543,0.924,0.023,0.972,0.061,0.181,0.003,0.432,0.505,0.593,0.725,0.031,0.492,0.142,0.222,0.286,0.064,0.9,0.187,0.36,0.413,0.974,0.27,0.17,0.235,0.833,-8.49686e+07,0.76,0.896,0.667,0.285,0.55,0.14,0.694,0.695,0.624,0.019,0.125,0.576,0.694,0.658,0.302,0.371,0.466,0.678,0.593,0.851,0.484,0.018,0.464,0.119,0.152,0.8,0.087,0.06,0.926,0.01,0.757,0.17,0.315,0.576,0.227,0.043,0.758,0.164,0.109,0.882,0.086,0.565,0.487,0.577,0.474,0.625,0.627,0.629,0.928,0.423,0.52,0.902,0.962,0.123,0.596,0.737,0.261,0.195,0.525,0.264,0.26,0.202,0.116,0.03,0.326,0.011,0.771,0.411,0.547,0.153,0.52,0.79,0.924,0.188,0.763,0.94,0.851,0.662,0.829,0.9,0.713,0.958,0.578,0.365,0.007,0.477,0.2,0.058,0.439,0.303,0.76,0.357,0.324,0.477,0.108,0.113,0.887,0.801,0.85,0.46,0.428,0.993,0.384,0.405,0.54,0.111,0.704,0.835,0.356,0.072,0.35,0.823,0.485,0.556,0.216,0.626,0.357,0.526,0.357,0.337,0.271,0.869,0.361,0.896,0.022,0.617,0.112,0.717,0.696,0.585,0.041,0.423,0.129,0.229,0.565,0.559,0.932,0.296,0.855,0.053,0.962,0.584,0.734,-4.46376e+06,0.972,0.457,0.369,0.532,0.963,0.607,0.483,0.911,0.635,0.067,0.848,0.675,0.938,0.223,0.142,0.754,0.511,0.741,0.175,0.459,0.825,0.221,0.87,0.626,0.934,0.205,0.783,0.85,0.398,0.279,0.701,0.193,0.734,0.637,0.534,0.556,0.993,0.176,0.705,0.962,0.548,0.881,0.3,0.413,0.641,0.855,0.855,0.142,0.462,0.611,0.877,0.424,0.678,0.752,0.443,0.296,0.673,0.04,0.313,0.875,0.072,0.818,0.61,0.017,0.932,0.112,0.695,0.169,0.831,0.04,0.488,0.685,0.09,0.497,0.589,0.99,0.145,0.353,0.314,0.651,0.74,0.044,0.258,0.335,0.759,0.192,0.605,0.264,0.181,0.503,0.829,0.775,0.608,0.292,0.997,0.549,0.556,0.561,0.627,0.467,0.541,0.129,0.24,0.813,0.174,0.601,0.077,0.215,0.683,0.213,0.992,0.824,0.601,0.392,0.759,0.67,0.428,0.027,0.084,0.075,0.786,0.498,0.97,0.287,0.847,0.604,0.503,0.221,0.663,0.706,0.363,0.01,0.171,0.489,0.24,0.164,0.542,0.619,0.913,0.591,0.704,0.818,0.232,0.75,0.205,0.975,0.539,0.303,0.422,0.098,0.247,0.584,0.648,0.971,0.864,0.913,0.075,0.545,0.712,0.546,0.678,0.769,0.262,0.519,0.985,0.289,0.944,0.865,0.54,0.245,0.508,0.318,0.87,0.601,0.323,0.132,0.472,0.152,0.087,0.57,0.763,0.901,0.103,0.423,0.527,0.6,0.969,0.015,0.565,0.028,0.543,0.347,0.088,0.943,0.637,0.409,0.463,0.049,0.681,0.588,0.342,0.608,0.06,0.221,0.758,0.954,0.888,0.146,0.69,0.949,0.843,0.43,0.62,0.748,0.067,0.536,0.783,0.035,0.226,0.185,0.038,0.853,0.629,0.224,0.748,0.923,0.359,0.257,0.766,0.944,0.955,0.318,0.726,0.411,0.025,0.355,0.001,0.549,0.496,-8.50092e+07,0.515,0.964,0.342,0.075,0.913,0.142,0.196,0.948,0.072,0.426,0.606,0.173,0.429,0.404,0.705,0.626,0.812,0.375,0.093,0.565,0.036,0.736,0.141,0.814,0.994,0.256,0.652,0.936,0.838,0.482,0.355,0.015,0.131,0.23,0.841,0.625,0.011,0.637,0.186,0.69,0.65,0.662,0.634,0.893,0.353,0.416,0.452,0.008,0.262,0.233,0.454,0.303,0.634,0.303,0.256,0.148,0.124,0.317,0.213,0.109,0.028,0.2,0.08,0.318,0.858,0.05,0.155,0.361,0.264,0.903,0.676,0.643,0.909,0.902,0.561,0.489,0.948,0.282,0.653,0.674,0.22,0.402,0.923,0.831,0.369,0.878,0.259,0.008,0.619,0.971,0.003,0.945,0.781,0.504,0.392,0.685,0.313,0.698,0.589,0.722,0.938,0.037,0.41,0.461,0.234,0.508,0.961,0.959,0.493,0.515,0.269,0.937,0.869,0.058,0.7,0.971,0.264,0.117,0.215,0.555,0.815,0.33,0.039,0.212,0.288,0.082,0.954,0.085,0.71,0.484,0.774,0.38,0.815,0.951,0.541,0.115,0.679,0.11,0.898,0.073,0.788,0.977,0.132,0.956,0.689,0.113,0.008,0.941,0.79,0.723,0.363,0.028,0.184,0.778,0.2,0.071,0.885,0.974,0.071,0.333,0.867,0.153,0.295,0.168,0.825,0.676,0.629,0.65,0.598,0.309,0.693,0.686,0.08,0.116,0.249,0.667,0.528,0.679,0.864,0.421,0.405,0.826,0.816,0.516,0.726,0.666,0.087,0.681,0.964,0.34,0.686,0.021,0.662,0.721,0.064,0.309,0.415,0.902,0.873,0.124,0.941,0.745,0.762,0.423,0.531,0.806,0.268,0.318,0.602,0.907,0.307,0.481,0.012,0.136,0.63,0.114,0.809,0.084,0.556,0.29,0.293,0.996,0.152,0.054,0.345,0.708,0.248,0.491,0.712,0.131,0.114,0.439,0.958,0.722,0.704,0.995,0.052,0.269,0.479,0.238,0.423,0.918,0.866,0.659,0.498,0.486,0.196,0.462,0.633,0.158,0.022,0.146,0.392,0.037,0.925,0.647,0.458,0.602,0.807,0.098,0.83,0.292,0.6,0.278,0.799,0.352,0.448,0.882,0.54,0.315,0.575,0.762,0.567,0.336,0.397,0.418,0.897,0.828,0.851,0.816,0.23,0.449,0.925,0.658,0.229,0.52,0.94,0.56,0.147,0.162,0.655,0.675,0.792,0.361,0.754,0.398,0.146,0.714,0.946,0.188,0.569,0.638,0.663,0.075,0.515,0.521,0.475,0.615,0.528,0.234,0.57,0.905,0.464,0.557,0.962,0.161,0.524,0.549,0.469,0.33,0.923,0.35,0.333,0.925,0.91,0.737,0.336,0.337,0.278,0.393,0.636,0.714,0.164,0.591,0.949,0.135,0.505,0.337,0.004,0.337,0.623,0.664,0.97,0.608,0.568,0.281,0.085,0.152,0.373,0.652,0.194,0.876,0.826,0.396,0.572,0.249,0.64,0.174,0.819,0.943,0.611,0.941,0.289,0.419,0.565,0.805,0.585,0.216,0.45,0.615,0.609,0.064,0.166,0.893,0.074,0.509,0.3,0.695,0.573,0.589,0.161,0.172,0.968,2.27637e+21,0.031,0.268,0.426,0.51,0.422,0.774,0.779,0.91,0.552,0.182,0.391,0.495,0.764,0.874,0.364,0.902,0.255,0.46,0.474,0.972,0.821,0.122,0.547,0.577,0.789,0.605,0.195,0.594,0.95,0.343,0.754,0.481,0.012,0.672,0.439,0.428,0.912,0.762,0.967,0.408,0.415,0.908,0.223,0.759,0.434,0.204,0.486,0.319,0.958,0.945,0.806,0.166,0.7,0.367,0.692,0.787,0.532,0.556,0.974,0.447,0.021,0.283,0.222,0.331,0.376,0.583,0.948,0.723,0.982,0.018,0.776,0.22,0.111,0.182,0.856,0.49,0.925,0.324,0.486,0.677,0.969,0.643,0.534,0.677,0.668,0.068,0.991,0.196,0.783,0.828,0.727,0.426,0.871,0.697,0.612,0.703,0.027,0.408,0.545,0.508,0.185,0.238,0.237,0.443,0.313,0.501,0.85,0.128,0.111,0.65,0.149,0.192,0.454,0.869,0.681,0.465,0.267,0.713,0.793,0.634,0.472,0.972,0.83,0.901,0.442,0.177,0.877,0.77,0.702,0.364,0.381,0.59,0.823,0.237,0.023,0.179,0.595,0.169,0.327,0.042,0.31,0.182,0.058,0.926,0.487,0.67,0.528,0.651,0.258,0.213,0.86,0.783,0.286,0.742,0.61,0.472,0.128,0.434,0.841,0.718,0.503,0.867,0.865,0.938,0.881,0.257,0.75,0.614,0.598,0.458,0.661,0.063,0.756,0.807,0.278,0.489,0.435,0.365,0.075,0.586,0.386,0.833,0.36,0.33,0.048,0.928,0.492,0.433,0.84,0.766,0.735,0.81,0.599,0.837,0.892,0.982,0.328,0.352,0.369,0.244,0.794,0.608,0.252,0.647,0.432,0.535,0.208,0.264,0.497,0.243,0.649,0.015,0.841,0.189,0.1,0.812,0.648,0.523,0.851,0.474,0.633,0.891,0.2,0.854,0.99,0.697,0.919,0.78,0.578,0.931,0.544,0.34,0.487,0.899,0.525,0.483,0.538,0.492,0.193,0.252,0.011,0.56,0.834,0.84,0.497,0.785,0.529,0.54,0.805,0.791,0.392,0.21,0.549,0.578,0.979,0.971,0.277,0.073,0.193,0.62,0.497,0.826,0.276,0.79,0.582,0.578,0.159,0.418,0.489,0.159,0.449,0.924,0.072,0.38,0.008,0.967,0.208,0.477,0.503,-8.55288,-9.61591,-8.10317,-10.3179,-9.95866,-9.89553,-10.7859,-11.0219,-10.2998,-7.6948,0.163,0.683,0.716,0.932,0.452,0.741,0.954,0.813,0.862,0.396,0.46,0.615,0.904,0.599,0.548132,0.487767,0.233,0.24633,0.0623203,0.255589,0.255727,0.255704,0.255937,0.453695
	};
	for(int i=0;i<1522;i++){
		data[i]=my_weights[i];
	}
}