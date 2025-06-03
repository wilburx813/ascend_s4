
// #include "pows_tiling.h"
// #include "register/op_def_registry.h"

// namespace optiling {
// static ge::graphStatus TilingFunc(gert::TilingContext* context)
// {

//   PowsTilingData tiling;
//   const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
//   context->SetBlockDim(8);
//   tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
//   context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

//   return ge::GRAPH_SUCCESS;
// }
// }

#include<vector>
#include "pows_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"          
using namespace std;                                                              
namespace optiling {                              
const uint32_t BLOCK_SIZE = 32;
void printVector(vector<uint32_t> &x_shape){
    for(int i = 0; i < x_shape.size(); ++i){
        printf("%d ", x_shape[i]);
    }
}

void printInts(uint32_t *ints, int32_t length){
    for(int i = 0; i < length; ++i){
        printf("%d ", ints[i]);
    }
}             
static ge::graphStatus TilingFunc(gert::TilingContext* context) {  
    //TODO      
    PowsTilingData tiling;  
    int32_t NUM = 8;   
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);  
    auto aivNum = ascendcPlatform.GetCoreNum();


    uint32_t input_num=2;                  
    // uint32_t shapeInf[8] = {};  
    uint32_t inputLength[2] = {};   
    // uint32_t length = 0;        
    // for (int i = 0; i < input_num; ++i) 
    //     length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < input_num; ++i) { 
        // const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();       
        // shapeInf[i*4]=shape->GetStorageShape().GetDimNum();           
        // for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {    
            // shapeInf[i*4+j] = shape->GetStorageShape().GetDim(j-1);                   
        // } 
    }               
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }             
    // bool boardCast = 0;    
    if (inputLength[0] != total_length ||           
        inputLength[1] != total_length ) {    
        // boardCast = 1;
        //TODO:shapeInf放到这
        context->SetTilingKey(1); 
        int32_t y_ndarray[20], x1_ndarray[20], x2_ndarray[20];
        int32_t y_dimensional, x1_dimensional, x2_dimensional;
        auto shape_y = context->GetOutputShape(0)->GetOriginShape();
        auto shape_x1 = context->GetInputTensor(0)->GetOriginShape();
        auto shape_x2 = context->GetInputTensor(1)->GetOriginShape();

        y_dimensional = shape_y.GetDimNum();
        x1_dimensional = shape_x1.GetDimNum();
        x2_dimensional = shape_x2.GetDimNum();

        int32_t max_y_dimensional;
        max_y_dimensional = y_dimensional;
        if (x1_dimensional > max_y_dimensional)
            max_y_dimensional = x1_dimensional;
        if (x2_dimensional > max_y_dimensional)
            max_y_dimensional = x2_dimensional;

        for (int i = 0; i < max_y_dimensional; i++)
        {
            if (i < y_dimensional)
                y_ndarray[y_dimensional - i - 1] = shape_y.GetDim(i);
            else
                y_ndarray[i] = 1;
            if (i < x1_dimensional)
                x1_ndarray[x1_dimensional - i - 1] = shape_x1.GetDim(i);
            else
                x1_ndarray[i] = 1;
            if (i < x2_dimensional)
                x2_ndarray[x2_dimensional - i - 1] = shape_x2.GetDim(i);
            else
                x2_ndarray[i] = 1;
        }
        tiling.set_y_dimensional(max_y_dimensional);
        tiling.set_y_ndarray(y_ndarray);
        tiling.set_x1_ndarray(x1_ndarray);
        tiling.set_x2_ndarray(x2_ndarray);

        int32_t y_sumndarray[20], x1_sumndarray[20], x2_sumndarray[20];
        y_sumndarray[0] = 1;
        x1_sumndarray[0] = 1;
        x2_sumndarray[0] = 1;
        for (int i = 1; i <= max_y_dimensional; i++)
        {
            y_sumndarray[i] = y_sumndarray[i - 1] * y_ndarray[i - 1];
            x1_sumndarray[i] = x1_sumndarray[i - 1] * x1_ndarray[i - 1];
            x2_sumndarray[i] = x2_sumndarray[i - 1] * x2_ndarray[i - 1];
        }
        tiling.set_y_sumndarray(y_sumndarray);
        tiling.set_x1_sumndarray(x1_sumndarray);
        tiling.set_x2_sumndarray(x2_sumndarray);
        printf("=== Debugging Tiling Parameters ===\n");
        printf("y_dimensional: %d\n", y_dimensional);

        printf("y_ndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", y_ndarray[i]);
        }
        printf("\n");

        printf("x1_ndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", x1_ndarray[i]);
        }
        printf("\n");

        printf("x2_ndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", x2_ndarray[i]);
        }
        printf("\n");

        printf("y_sumndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", y_sumndarray[i]);
        }
        printf("\n");

        printf("x1_sumndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", x1_sumndarray[i]);
        }
        printf("\n");

        printf("x2_sumndarray: ");
        for (int i = 0; i < y_dimensional; i++) {
            printf("%d ", x2_sumndarray[i]);
        }
        printf("\n");    
    }else{
        context->SetTilingKey(0);
        // printf("sizeoddatatype: %d",sizeofdatatype);    
        auto dt = context->GetInputTensor(0)->GetDataType();
        uint32_t sizeofdatatype;  
        if (dt == ge::DT_FLOAT) { 
            sizeofdatatype = 4;
            NUM = 6;
        }         
        else {      
            sizeofdatatype = 2;
            NUM = 10;               
        }           
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;                   
        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 1) / NUM;          
        tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
        uint32_t block_size = tiling_size * ALIGN_NUM;
        aivNum = 1;   
        uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
        uint32_t core_remain = total_length - aivNum * core_size;
        tiling.set_ALIGN_NUM(ALIGN_NUM);        
        tiling.set_block_size(block_size);            
        tiling.set_core_size(core_size);        
        tiling.set_core_remain(core_remain);    
    }
    // bool isTs=false;  
    // if(inputLength[0]>500000 ||    
    //    inputLength[1]>500000 ){           
    //     isTs=true;          
    //     context->SetTilingKey(0);              
    // }else{
    //     context->SetTilingKey(1);       
    // }
    
    // NUM =20;
    
    // tiling.set_shapeInf(shapeInf);          
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}                       
} 

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Pows : public OpDef {
public:
    explicit Pows(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Pows);
}
