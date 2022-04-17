#include "lenet.h"

int main(){
    float photo[10],std[10];//photo为当前输入。std为训练时的标准输出，用于和预测输出进行比对，然后校正权重参数
    float data[1522];//1512位为权重，最后10位用于保存当前输入
	float output[10];//对应的预测输出
    float fc1[48],fc2[24];//全连接层的输出结果，用于校正权重参数
    float **data_input = new float*[873];//真实的输入，用于存储噪声数据，共873组，每组为1*10的向量
    for(int i=0;i<873;i++)
	    data_input[i] = new float[10];
    float data_output[873][10];//存储预测的结果，同时输出到txt文件
    float ans=0,res=0;//这两个参数用于预测阶段，ans保存每组输入与预测误差函数值，res为记录平均误差函数值
    srand((unsigned)time(NULL));

    //read weights
    read_weights(data);

    float A=30;
    float B=10;     //A和B是S型函数的参数
    float ETA_W=9.8;   	  //权值调整率
    float ETA_B=3.3;	  //阀值调整率

    float best_ETA_W,best_ETA_B,best_A,best_B;
    //超参数训练 ETA_W=9.8 ETA_B=3.3
    //trian_ETA_W_B(&best_ETA_W,&best_ETA_B,photo,data,output,fc2,fc1,std,A,B);
    //cout <<best_ETA_W <<" " <<best_ETA_B <<endl;
    //超参数训练 A=1 B=6
    //trian_sigmoid_A_B(&best_A,&best_B,photo,data,output,fc2,fc1,std,ETA_W,ETA_B);
    //cout <<best_A <<" " <<best_B <<endl;


    //data input = 873*10
    for(int loop=0;loop<25;loop++){
        res=0;ans=0;
        read_weights(data);//读取已有权重
        for(int i=0;i<10;i++)
            data_output[0][i]=0;
        
        switch(loop){//共25个文件，每次训练一个文件的数据，每个文件都是873组，每组1*10的矩阵输入
            case 0:get_input_1_02_60(data_input);break;
            case 1:get_input_1_02_70(data_input);break;
            case 2:get_input_1_02_80(data_input);break;
            case 3:get_input_1_02_90(data_input);break;
            case 4:get_input_1_02_100(data_input);break;
            case 5:get_input_1_04_60(data_input);break;
            case 6:get_input_1_04_70(data_input);break;
            case 7:get_input_1_04_80(data_input);break;
            case 8:get_input_1_04_90(data_input);break;
            case 9:get_input_1_04_100(data_input);break;
            case 10:get_input_1_06_60(data_input);break;
            case 11:get_input_1_06_70(data_input);break;
            case 12:get_input_1_06_80(data_input);break;
            case 13:get_input_1_06_90(data_input);break;
            case 14:get_input_1_06_100(data_input);break;
            case 15:get_input_1_08_60(data_input);break;
            case 16:get_input_1_08_70(data_input);break;
            case 17:get_input_1_08_80(data_input);break;
            case 18:get_input_1_08_90(data_input);break;
            case 19:get_input_1_08_100(data_input);break;
            case 20:get_input_1_10_60(data_input);break;
            case 21:get_input_1_10_70(data_input);break;
            case 22:get_input_1_10_80(data_input);break;
            case 23:get_input_1_10_90(data_input);break;
            case 24:get_input_1_10_100(data_input);break;
            default:break;
        }
        
        //train 643
        for(int i=0;i<643;i++){
            for(int loop1_i=0;loop1_i<10;loop1_i++){
                data[loop1_i+1512]=data_input[i][loop1_i];//获取当前的输入，储存到data的最后10位
                std[loop1_i]=data_input[i+1][loop1_i];//获取当前输入的真实输出（即下一组数据
            }
            LetNet(photo,data,output,fc1,fc2);//前向传播
            for(int j=0;j<10;j++){
                data_output[i+1][j]=output[j];//保存预测结果
                //计算训练误差
                //ans+=0.5*(data_input[i+1][j]-output[j])*(data_input[i+1][j]-output[j]); 
            }
            //cout << ans <<" ";
            //res+=ans;
            //ans=0;
            feedback(std,data,output,fc1,fc2,ETA_W,ETA_B,A,B);//反向传播
        }
        //cout <<endl <<endl <<"res:" <<res/643 <<endl;
        res=0;
        //test 230
        for(int loop1_i=0;loop1_i<10;loop1_i++){
            data[loop1_i+1512]=data_input[643][loop1_i];//获取当前的输入，储存到data的最后10位
        }
        /*
        for(int x=0;x<10;x++){
            cout <<data_input[643][x] <<" ";
        }cout <<endl;
        for(int x=0;x<10;x++){
            cout <<output[x] <<" ";
        }cout <<endl;
        */
        LetNet(photo,data,output,fc1,fc2);//前向传播 
        feedback(std,data,output,fc1,fc2,ETA_W,ETA_B,A,B);//反向传播
        for(int i=644;i<873;i++){
            for(int j=0;j<10;j++){
                data_output[i][j]=output[j];//保存预测结果
                data[j+1512]=output[j];//将当前预测结果，作为下一次预测的输入
                std[j]=data_input[i][j];//获取当前输入的真实输出（即下一组数据
                //计算预测误差
                ans+=0.5*(data_input[i][j]-output[j])*(data_input[i][j]-output[j]);
            }
            //cout << ans <<" ";
            /*
            for(int x=0;x<10;x++){
                cout <<data[x+1512] <<" ";
            }cout <<endl;
            */
            res+=ans;
            ans=0;
            LetNet(photo,data,output,fc1,fc2);//前向传播 
            feedback(std,data,output,fc1,fc2,ETA_W,ETA_B,A,B);//反向传播
            /*
            for(int x=0;x<10;x++){
                cout <<output[x] <<" ";
            }cout <<endl;
            */
        }
        //cout <<endl <<endl <<"res:" <<res/230 <<endl;
        
        //每次预测完，将误差函数值输出，同时将预测结果写入文件
        string out_1,out_2;
        switch(loop%5){
            case 0:out_2="60";break;
            case 1:out_2="70";break;
            case 2:out_2="80";break;
            case 3:out_2="90";break;
            case 4:out_2="100";break;
            default:break;
        }
        switch(loop/5){
            case 0:out_1="0.2";break;
            case 1:out_1="0.4";break;
            case 2:out_1="0.6";break;
            case 3:out_1="0.8";break;
            case 4:out_1="1.0";break;
            default:break;
        }
        cout <<out_1 <<"_" <<out_2 <<":" <<"res:" <<res/230 <<endl;
        out_to_file(data_output,loop);
    }


    return 0;
}
