// #include "kernel_operator.h"
#define K_MAX_SHAPE_DIM 0     
#include "kernel_operator.h"
using namespace AscendC;     
using namespace std;
constexpr int32_t BUFFER_NUM = 2;      
// extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
//     GET_TILING_DATA(tiling_data, tiling);
//     // TODO: user kernel impl
// }

class KernelPows {          
public:                            
    __aicore__ inline KernelPows() {}  
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, 
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {                      
        this->blockLength = core_size + core_remain;  
        this->tileLength = block_size;                                                            
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        // printf("[DEBUG] core_size = %lu, core_remain = %lu\n", core_size, core_remain);
        // this->blockLength = core_size + core_remain;
        // printf("[DEBUG] Initial blockLength = %lu\n", this->blockLength);

        // printf("[DEBUG] tileLength (block_size) = %lu\n", block_size);
        // this->tileLength = block_size;

        // size_t alignment_adjustment = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        // printf("[DEBUG] ALIGN_NUM = %lu, alignment_adjustment = %lu\n", ALIGN_NUM, alignment_adjustment);

        // this->blockLength = this->blockLength + alignment_adjustment;
        // printf("[DEBUG] Aligned blockLength = %lu\n", this->blockLength);

        // size_t tiles_needed = this->blockLength / this->tileLength;
        // size_t remainder = this->blockLength % this->tileLength;
        // printf("[DEBUG] tiles_needed = %lu, remainder = %lu\n", tiles_needed, remainder);

        // this->tileNum = tiles_needed + (remainder > 0);
        // printf("[DEBUG] Final tileNum = %lu\n", this->tileNum);

        Gm_x1.SetGlobalBuffer((__gm__ DTYPE_X1*)x1 , this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ DTYPE_X2*)x2 , this->blockLength);
        Gm_y.SetGlobalBuffer((__gm__ DTYPE_Y*)y , this->blockLength); 
       
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1)); 
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));   
        
        pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));   
        pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
         

    }    
    __aicore__ inline void Process() { 
        int32_t loopCount = this->tileNum;              
        for (int32_t i = 0; i < loopCount-1; i++) {   
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);  
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length); 
        CopyOut(loopCount - 1, length);    
    }
          
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_X1>  x1 = Q_x1.AllocTensor<DTYPE_X1>();    
        LocalTensor<DTYPE_X2>  x2 = Q_x2.AllocTensor<DTYPE_X2>();   
        DataCopy(x1, Gm_x1[progress * this->tileLength], length); 
        DataCopy(x2, Gm_x2[progress * this->tileLength], length); 
        Q_x1.EnQue(x1);      
        Q_x2.EnQue(x2);        
    } 
    __aicore__ inline void Compute(int32_t progress, uint32_t length) { 
        LocalTensor<DTYPE_X1> x1 = Q_x1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.DeQue<DTYPE_X2>();  
        LocalTensor<DTYPE_Y> y = Q_y.AllocTensor<DTYPE_Y>();     
        if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t>) {
            auto float_x1 = B_x1.Get<float>();     
            auto float_x2 = B_x2.Get<float>();      
            Cast(float_x1, x1, RoundMode::CAST_NONE, length); 
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);  
            
            Ln(float_x1, float_x1, length);
            Mul(float_x1, float_x1, float_x2, length);
            Exp(float_x1, float_x1, length);

            Cast(y, float_x1, RoundMode::CAST_ROUND, length); 
        } if constexpr (std::is_same_v<DTYPE_X1, half>) {
            auto float_x1 = B_x1.Get<float>();     
            auto float_x2 = B_x2.Get<float>();      
            Cast(float_x1, x1, RoundMode::CAST_NONE, length); 
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);  
            
            Ln(float_x1, float_x1, length);
            Mul(float_x1, float_x1, float_x2, length);
            Exp(float_x1, float_x1, length);

            Cast(y, float_x1, RoundMode::CAST_ROUND, length);
        }else{     
            Ln(x1, x1, length);
            Mul(x1, x1, x2, length);
            Exp(y, x1, length);        
        }       
        Q_x1.FreeTensor(x1); 
        Q_x2.FreeTensor(x2); 
        Q_y.EnQue<DTYPE_Y>(y); 
    }   
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<DTYPE_Y> y = Q_y.DeQue<DTYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    } 
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2; 
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    GlobalTensor<DTYPE_X1> Gm_x1; 
    GlobalTensor<DTYPE_X2> Gm_x2;  
    GlobalTensor<DTYPE_Y> Gm_y; 
    uint32_t blockLength;  
    uint32_t tileNum;
    uint32_t tileLength;    
}; 


template<typename TYPE_X1, typename TYPE_X2,  typename TYPE_Y> class KernelPows_Broadcast {
    using T = TYPE_Y;    
public:
    __aicore__ inline KernelPows_Broadcast() {}     
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2,GM_ADDR y,  
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {    
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!"); 
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
        // auto startPointer = core_size * GetBlockIdx();  
        auto bufferlength = this->blockLength; 
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, bufferlength);  
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, bufferlength);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, bufferlength);
        printf("bufferlength: %d",bufferlength);
        // if(is_same_v<DTYPE_X1,half>){
            pipe.InitBuffer(tmp32Buffer1, 1 * sizeof(half));
            pipe.InitBuffer(tmp32Buffer2, 1 * sizeof(half)); 
        // }
        pipe.InitBuffer(tmp1Buffer, 1 * sizeof(float));   
        pipe.InitBuffer(tmp2Buffer, 1 * sizeof(float));   
        // pipe.InitBuffer(tmp32Buffer, 1 * sizeof(float));    
    
    }    
    __aicore__ inline void Process(uint32_t shapeInf[2*4]) {    
        LocalTensor<float> tmp1 = tmp1Buffer.Get<float>();     
        LocalTensor<float> tmp2 = tmp2Buffer.Get<float>(); 
        // if constexpr(is_same_v<TYPE_X1,half>){
            LocalTensor<TYPE_Y> tmp3 = tmp32Buffer1.Get<TYPE_Y>();     
            LocalTensor<TYPE_Y> tmp4 = tmp32Buffer2.Get<TYPE_Y>(); 
        // }
        uint32_t input_num=2;          
        int max_dim=0;  
        for(int i=0;i<input_num;i++){ 
            if(shapeInf[i*4+0]>max_dim){  
                max_dim = shapeInf[i*4+0]; 
            }   
        }    
        if (max_dim == 1) {
            printf("max_dim = 1");
            int max_index = 0;   
            for (int i = 0; i < input_num; i++) {
                if (shapeInf[i * 4 + 1] > max_index) {
                    max_index = shapeInf[i * 4 + 1];
                }       
            } 
            for (int i = 0; i < max_index; i++) {     
                int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i;  
                int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i;  
                if constexpr (std::is_same_v<T, float>){
                    float x1_value  = static_cast<float>(Gm_x1(index_x1_i));  
                    float x2_value  = static_cast<float>(Gm_x2(index_x2_i)); 
                    Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                    Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                    Ln(tmp1, tmp1, 1);
                    Mul(tmp2, tmp1, tmp2, 1);
                    Exp(tmp2, tmp2, 1);         
                    Gm_y(i) = static_cast<TYPE_Y>(tmp2(0));       
                }else if constexpr(is_same_v<TYPE_X1,half>){
                            
                    // half x1_value  = Gm_x1(index_x1_i);  
                    // half x2_value  = Gm_x2(index_x2_i); 
                    // printf("test\n");
                    // Duplicate<TYPE_X1>(tmp1, static_cast<TYPE_X1>(x1_value), 1);    
                    // Duplicate<TYPE_X1>(tmp2, static_cast<TYPE_X1>(x2_value), 1);
                    // Cast(tmp3, tmp1, RoundMode::CAST_NONE, 1); 
                    // Cast(tmp4, tmp2, RoundMode::CAST_NONE, 1); 
                    // Ln(tmp3, tmp3, 1);
                    // Mul(tmp4, tmp3, tmp4, 1);
                    // Exp(tmp4, tmp4, 1);    
                    // Cast(tmp2, tmp4, RoundMode::CAST_ROUND, 1);     
                    // Gm_y(i) = static_cast<TYPE_X1>(tmp2(0));  
                }

            }     
        } 
        else if (max_dim == 2) {  
            printf("max_dim = 2");
            int max_index[2] = {};  
            for (int i = 0; i < input_num; i++) { 
                for (int j = 1; j <= shapeInf[i * 4 + 0]; j++) {  
                    if (shapeInf[i * 4 + j] > max_index[j - 1]) {
                        max_index[j - 1] = shapeInf[i * 4 + j];
                    }
                }  
            } 
            for (int i = 0; i < max_index[0]; i++) {           
                for (int j = 0; j < max_index[1]; j++) {   
                    int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i; 
                    int index_x1_j = (shapeInf[0 * 4 + 2] <= 1) ? 0 : j; 
                    int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i; 
                    int index_x2_j = (shapeInf[1 * 4 + 2] <= 1) ? 0 : j;  
                    if constexpr (std::is_same_v<T, float>){
                        float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * shapeInf[0 * 4 + 2] + index_x1_j));   
                        float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * shapeInf[1 * 4 + 2] + index_x2_j));  
                        Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                        Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                        Ln(tmp1, tmp1, 1);
                        Mul(tmp2, tmp1, tmp2, 1);
                        Exp(tmp2, tmp2, 1);         
                        Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp2(0));     
                    }else if constexpr(is_same_v<TYPE_X1,half>){
                            
                        // half x1_value  = Gm_x1(index_x1_i);  
                        // half x2_value  = Gm_x2(index_x2_i); 
                        // printf("test\n");
                        // Duplicate<TYPE_X1>(tmp1, static_cast<TYPE_X1>(x1_value), 1);    
                        // Duplicate<TYPE_X1>(tmp2, static_cast<TYPE_X1>(x2_value), 1);
                        // Cast(tmp3, tmp1, RoundMode::CAST_NONE, 1); 
                        // Cast(tmp4, tmp2, RoundMode::CAST_NONE, 1); 
                        // Ln(tmp3, tmp3, 1);
                        // Mul(tmp4, tmp3, tmp4, 1);
                        // Exp(tmp4, tmp4, 1);    
                        // Cast(tmp2, tmp4, RoundMode::CAST_ROUND, 1);     
                        // Gm_y(i) = static_cast<TYPE_X1>(tmp2(0));  
                    }
    
                }   
            }     
        }     
        else if (max_dim == 3){    
            int max_index[3]={}; 
            printf("max_dim=3");
            for(int i=0;i<input_num;i++){    
                for (int j = 1; j <= shapeInf[i*4+0]; j++) {  
                    if(shapeInf[i*4+j]>max_index[j-1]){
                        max_index[j-1] = shapeInf[i*4+j];
                    } 
                }     
            }  
            for (int i = 0; i < max_index[0]; i++) {            
                for (int j = 0; j < max_index[1]; j++) {   
                    for (int k = 0; k < max_index[2]; k++) {          
                        int index_x1_i = (shapeInf[0 * 4 + 1] <= 1) ? 0 : i;  
                        int index_x1_j = (shapeInf[0 * 4 + 2] <= 1) ? 0 : j; 
                        int index_x1_k = (shapeInf[0 * 4 + 3] <= 1) ? 0 : k; 
                        int index_x2_i = (shapeInf[1 * 4 + 1] <= 1) ? 0 : i;  
                        int index_x2_j = (shapeInf[1 * 4 + 2] <= 1) ? 0 : j;  
                        int index_x2_k = (shapeInf[1 * 4 + 3] <= 1) ? 0 : k;  
                        if constexpr (std::is_same_v<T, float>){
                            float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * (shapeInf[0*4+2] * shapeInf[0*4+3]) + index_x1_j  * shapeInf[0*4+3] + index_x1_k));  
                            float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * (shapeInf[1*4+2] * shapeInf[1*4+3]) + index_x2_j  * shapeInf[1*4+3] + index_x2_k));  
                            // printf("%f, %f",x1_value,x2_value);
                            printf("float, x1_value = %f, x2_value = %f\n", x1_value, x2_value);
                            Duplicate<TYPE_Y>(tmp1, static_cast<TYPE_Y>(x1_value), 1);    
                            Duplicate<TYPE_Y>(tmp2, static_cast<TYPE_Y>(x2_value), 1);    
                            // Div(tmp2, tmp1, tmp2, 1); 
                            Ln(tmp1, tmp1, 1);
                            Mul(tmp2, tmp1, tmp2, 1);
                            Exp(tmp2, tmp2, 1);        
                            Gm_y(i * (max_index[1] * max_index[2]) + j * max_index[2] + k) = static_cast<TYPE_Y>(tmp2(0));     
                        }else if constexpr(is_same_v<TYPE_X1,half>){
                            
                            float x1_value  = static_cast<float>(Gm_x1 (index_x1_i  * (shapeInf[0*4+2] * shapeInf[0*4+3]) + index_x1_j  * shapeInf[0*4+3] + index_x1_k));  
                            float x2_value  = static_cast<float>(Gm_x2 (index_x2_i  * (shapeInf[1*4+2] * shapeInf[1*4+3]) + index_x2_j  * shapeInf[1*4+3] + index_x2_k)); 
                            printf("half, x1_value = %f, x2_value = %f\n", x1_value, x2_value);  
                            Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);    
                            Duplicate<float>(tmp2, static_cast<float>(x2_value), 1);    
                            Ln(tmp1, tmp1, 1);
                            Mul(tmp2, tmp1, tmp2, 1);
                            Exp(tmp2, tmp2, 1);         
                            Gm_y(i * max_index[1] + j) = static_cast<TYPE_Y>(tmp2(0));  
                        }
        
                    }   
                }   
            }              
        }     
    }      
private:
    TPipe pipe;    
    GlobalTensor<TYPE_X1> Gm_x1;     
    GlobalTensor<TYPE_X2> Gm_x2;  
    GlobalTensor<TYPE_Y> Gm_y;  
    TBuf<QuePosition::VECCALC> tmp1Buffer,tmp2Buffer,tmp32Buffer1,tmp32Buffer2; 
    uint32_t blockLength;   
};  

template <typename T>
class KernelPowsBroadCast
{
    using Tfp = float;

public:
    __aicore__ inline KernelPowsBroadCast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                int32_t y_dimensional,
                                int32_t *y_ndarray, int32_t *x1_ndarray, int32_t *x2_ndarray,
                                int32_t *y_sumndarray, int32_t *x1_sumndarray, int32_t *x2_sumndarray)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->y_dimensional = y_dimensional;

        this->y_ndarray = y_ndarray;
        this->x1_ndarray = x1_ndarray;
        this->x2_ndarray = x2_ndarray;
        this->y_sumndarray = y_sumndarray;
        this->x1_sumndarray = x1_sumndarray;
        this->x2_sumndarray = x2_sumndarray;

        x1Gm.SetGlobalBuffer((__gm__ T *)x1, 1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2, 1);
        yGm.SetGlobalBuffer((__gm__ T *)y, 1);
        pipe.InitBuffer(tmp32Buffer1, 1 * sizeof(half));
        pipe.InitBuffer(tmp32Buffer2, 1 * sizeof(half)); 
        pipe.InitBuffer(tmp1Buffer, 1 * sizeof(float));   
        pipe.InitBuffer(tmp2Buffer, 1 * sizeof(float));   
    }
    __aicore__ inline void Process()
    {
        Tfp x1, x2;
        int dim = this->y_dimensional;
        LocalTensor<float> tmp1 = tmp1Buffer.Get<float>();     
        LocalTensor<float> tmp2 = tmp2Buffer.Get<float>(); 

        LocalTensor<DTYPE_Y> tmp3 = tmp32Buffer1.Get<DTYPE_Y>();     
        LocalTensor<DTYPE_Y> tmp4 = tmp32Buffer2.Get<DTYPE_Y>(); 
        for (int j = 0; j < this->y_sumndarray[dim]; j++)
        {
            int x1_start = 0, x2_start = 0;
            for (int k = 0; k < dim; k++)
            {
                if (this->x1_ndarray[k] != 1)
                {
                    x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
                if (this->x2_ndarray[k] != 1)
                {
                    x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
                }
            }
            x1 = static_cast<Tfp>(x1Gm.GetValue(x1_start));
            x2 = static_cast<Tfp>(x2Gm.GetValue(x2_start));
            
            Duplicate<float>(tmp1, static_cast<float>(x1), 1);    
            Duplicate<float>(tmp2, static_cast<float>(x2), 1);   

            Ln(tmp1, tmp1, 1);
            Mul(tmp1, tmp1, tmp2, 1);
            Exp(tmp1, tmp1, 1);
            yGm.SetValue(j, tmp1(0));
        }
        // // this->x1_sumndarray[0] = 1;
        // // this->x1_sumndarray[1] = 5;
        // // this->x1_sumndarray[2] = 5;
        // for (int j = 0; j < this->y_sumndarray[dim]; j++)
        // {   
        //     int x1_start = 0, x2_start = 0;
        //     printf("\n--- j = %d ---\n", j);  // 外层循环变量
            
        //     for (int k = 0; k < dim; k++)
        //     {
        //         printf(
        //             "k=%d | x1_ndarray[%d]=%d | x1_sumndarray[%d]=%d | "
        //             "(j/y_sumndarray[k])=%d | (j/y_sumndarray[k]%%y_ndarray[k])=%d | "
        //             "x1_start += %d\n",
        //             k, k, this->x1_ndarray[k], k, this->x1_sumndarray[k],
        //             (j / this->y_sumndarray[k]),
        //             (j / this->y_sumndarray[k] % this->y_ndarray[k]),
        //             this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k])
        //         );
        //         printf(
        //             "k=%d | x2_ndarray[%d]=%d | x2_sumndarray[%d]=%d | "
        //             "(j/y_sumndarray[k])=%d | (j/y_sumndarray[k]%%y_ndarray[k])=%d | "
        //             "x2_start += %d\n",
        //             k, k, this->x2_ndarray[k], k, this->x2_sumndarray[k],
        //             (j / this->y_sumndarray[k]),
        //             (j / this->y_sumndarray[k] % this->y_ndarray[k]),
        //             this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k])
        //         );
                
        //         if (this->x1_ndarray[k] != 1)
        //         {
        //             x1_start += this->x1_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
        //         }
        //         if (this->x2_ndarray[k] != 1)
        //         {
        //             x2_start += this->x2_sumndarray[k] * (j / this->y_sumndarray[k] % this->y_ndarray[k]);
        //         }
        //     }
        
        //     printf("Final: x1_start = %d, x2_start = %d\n", x1_start, x2_start);
        //     if constexpr (std::is_same_v<T, float>){
        //         float x1_value  = static_cast<float>(x1Gm(x1_start));  
        //         float x2_value  = static_cast<float>(x2Gm(x2_start));  
        //         // printf("%f, %f",x1_value,x2_value);
        //         // printf("float, x1_start = %d, x2_start = %d\n", x1_start, x2_start);
        //         printf("float, x1_value = %f, x2_value = %f\n", x1_value, x2_value);
        //         Duplicate<float>(tmp1, static_cast<float>(x1_value), 1);    
        //         Duplicate<float>(tmp2, static_cast<float>(x2_value), 1);    
        //         // Div(tmp2, tmp1, tmp2, 1); 
        //         Ln(tmp1, tmp1, 1);
        //         Mul(tmp2, tmp1, tmp2, 1);
        //         Exp(tmp2, tmp2, 1);        
        //         yGm(j) = static_cast<DTYPE_Y>(tmp2(0));     
        //     }
        //     // x1 = static_cast<Tfp>(x1Gm.GetValue(x1_start));
        //     // x2 = static_cast<Tfp>(x2Gm.GetValue(x2_start));
        //     // yGm.SetValue(j, static_cast<T>(x1 / x2));
        // }
    }

private:
    TPipe pipe; 
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;

    int32_t y_dimensional;
    int32_t *y_ndarray;
    int32_t *x1_ndarray;
    int32_t *x2_ndarray;

    int32_t *y_sumndarray;
    int32_t *x1_sumndarray;
    int32_t *x2_sumndarray;
    TBuf<QuePosition::VECCALC> tmp1Buffer,tmp2Buffer,tmp32Buffer1,tmp32Buffer2; 
};

extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2,GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(0)){
        GET_TILING_DATA(tiling_data, tiling);    
        KernelPows op;  
        op.Init(x1, x2, y,
                tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        op.Process();    
        printf("KernelPows\n");
    }else if(TILING_KEY_IS(1)){
        GET_TILING_DATA(tiling_data, tiling);   
        printf("KernelPow_Broadcast\n");  

        // KernelPows_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;    
        // op.Init(x1, x2, y,      
        //     tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
        // op.Process(tiling_data.shapeInf);
        if(is_same_v<DTYPE_X1,float>){
            KernelPowsBroadCast<DTYPE_X1> op;
            op.Init(x1, x2, y,
                    tiling_data.y_dimensional,
                    tiling_data.y_ndarray, tiling_data.x1_ndarray, tiling_data.x2_ndarray,
                    tiling_data.y_sumndarray, tiling_data.x1_sumndarray, tiling_data.x2_sumndarray);
            op.Process();
        }   
        
          
    }
}   