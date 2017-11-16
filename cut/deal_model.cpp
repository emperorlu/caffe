#define CPU_ONLY
#include "caffe/caffe.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace caffe;

Phase phase = TEST;
enum BlobType {
	WeightsType = 0,
	BiasType
};

inline int  offset(const caffe::BlobProto * b,const int n,const int c = 0,const int h = 0,const int w = 0)
{
	return ((n * b->channels() +c)*b->height() + h) *b->width()+w;
}

// 参数为要裁剪的index 列表
// 效率为2*O(n) ,n为参数个数 
void transform_blob(BlobProto * ptr,
		const std::vector  <int> & num_list,
		const std::vector <int > & channel_list = std::vector<int>(0),
		const std::vector <int  > & height_list = std::vector<int>(0),
		const std::vector <int> & width_list = std::vector<int>(0))
{
	LOG(INFO) <<" in trasnform .." ;
    int num = ptr->num();
    int channel = ptr->channels();
    int width = ptr->width();
    int height = ptr->height();
    if(!(ptr->has_num() || ptr->has_channels() ||
    	ptr->has_height() || ptr->has_width()		))
	{
		LOG(INFO) <<"shape size" << ptr->shape().dim_size();
		if(ptr->shape().dim_size() == 4) // wights blob 一共4维
		{
			num = ptr->shape().dim(0);
			channel= ptr->shape().dim(1);
			height = ptr->shape().dim(2);
			width = ptr->shape().dim(3);
		}
		else if(ptr->shape().dim_size() == 1) // bais blob 只有一维
		{
			num = 1;
			channel = 1;
			height = 1;
			width = ptr->shape().dim(0);
		}
	}
    ::google::protobuf::RepeatedField< float >* data=  ptr->mutable_data();
    ::google::protobuf::RepeatedField< float >* new_data=new  ::google::protobuf::RepeatedField<float >();
    for(int n = 0;n<num;++n) if(find(num_list.begin(),num_list.end(),n)== num_list.end())
	{
		for(int c = 0;c<channel;++c) if(find(channel_list.begin(),channel_list.end(),c)== channel_list.end())
		{
			for(int h=0;h<height;++h)	if(find(height_list.begin(),height_list.end(),h)== height_list.end())
			{
				for(int w = 0;w<width;++w) if(find(width_list.begin(),width_list.end(),w)== width_list.end())
				{
					new_data->Add(data->Get(offset(ptr,n,c,h,w)));
				}
			}
		}
	}
	ptr->mutable_data()->Clear(); 
	ptr->mutable_data()->CopyFrom(*new_data);

    if(ptr->has_num() || ptr->has_channels() ||
    	ptr->has_height() || ptr->has_width()		)
	{
	ptr->set_num(num-num_list.size());
	ptr->set_channels(channel-channel_list.size());
	ptr->set_width(width-width_list.size());
	ptr->set_height(height-height_list.size());
	LOG(INFO) << "new shape : num : " <<ptr->num()<<"channels : "<<ptr->channels()
		<< "height : "<<ptr->height() <<"width : "<<ptr->width();
	LOG(INFO) <<"new size : "<<ptr->data().size();
	CHECK_EQ(ptr->data().size(), 
			ptr->num()*ptr->channels()*ptr->height()*ptr->width()
			);
	}
	else 
	{
		//ptr->mutable_shape()->mutable_dim()->Clear() ;
		LOG(INFO) <<"set new size ";
		if(ptr->shape().dim_size() ==4)
		{
			LOG(INFO) <<"dim size"<<ptr->shape().dim().size();
			ptr->mutable_shape()->mutable_dim()->Set(0,num-num_list.size()) ;
			ptr->mutable_shape()->mutable_dim()->Set(1,channel-channel_list.size());
			ptr->mutable_shape()->mutable_dim()->Set(2,height-height_list.size());
			ptr->mutable_shape()->mutable_dim()->Set(3,width-width_list.size());
		}
		else if(ptr->shape().dim_size() ==1)
		{
			LOG(INFO) <<"dim size"<<ptr->shape().dim().size();
			ptr->mutable_shape()->mutable_dim()->Set(0,width-width_list.size());
		}

	}
	LOG(INFO) <<"transform over..";

}
int cutLastLayerByL1(caffe::LayerParameter * currentLayer,caffe::LayerParameter  * nextLayer,float cutRate) //裁剪最后一个卷积层的时候需要同时减少第一个全连接层的参数
{
	LOG(INFO)<<"Cut current Layer type : "<<currentLayer->type()
		<<"name : "<<currentLayer->name();
	LOG(INFO)<<"Cut next Layer type : "<<nextLayer->type()
		<<"name : "<<nextLayer->name();

	caffe::BlobProto * wightsBlob = currentLayer->mutable_blobs(WeightsType);
	caffe::BlobProto * biasBlob = currentLayer->mutable_blobs(BiasType);

	int currentNum = wightsBlob->num();
	int currentChannels = wightsBlob->channels();
	int currentHeight = wightsBlob->height();
	int currentWidth = wightsBlob->width();
	if(!(wightsBlob->has_num() || wightsBlob->has_channels() ||
			wightsBlob->has_height() || wightsBlob->has_width()))
	{
		LOG(INFO) << "use shape .." ;
		currentNum = wightsBlob->shape().dim(0);
		currentChannels = wightsBlob->shape().dim(1);
		currentHeight = wightsBlob->shape().dim(2);
		currentWidth = wightsBlob->shape().dim(3);
	}

	
	CHECK_LT(cutRate,1.0)<< "cut Radio less than 1.0";
	int newNum = (1-cutRate)*currentNum;

	::google::protobuf::RepeatedField<float> * data = wightsBlob->mutable_data();
	std::vector <std::pair <float,int> > L1_list;
	for(int n = 0;n<currentNum;++n)  //计算每个过滤器的 绝对值之和
	{
		float abs_sum = 0;
		for(int c = 0;c<currentChannels;++c)
		{
			for(int h = 0;h<currentHeight;++h)
			{
				for(int w = 0;w<currentWidth;++w)
				{
					abs_sum += std::abs(data->Get(offset(wightsBlob,n,c,h,w)));
				}
			}
		}
		LOG(INFO) <<"abs_sum of index "<<n<<" : "<<abs_sum;
		L1_list.push_back(std::make_pair(abs_sum,n));
	}
	sort(L1_list.begin(),L1_list.end());  //排序
	std::vector <int> cut_num_list;
	for(int i =0;i<L1_list.size();++i) if(i < currentNum-newNum)
	{
		cut_num_list.push_back(L1_list[i].second);
		LOG(INFO)<<"cut index :"<<L1_list[i].second;
	}

	transform_blob(wightsBlob,cut_num_list); //裁剪操作
	transform_blob(biasBlob,std::vector<int>(0),std::vector<int>(0),std::vector<int>(0),cut_num_list); //bias 裁剪width


	caffe::BlobProto * nextWightsBlob = nextLayer->mutable_blobs(WeightsType);
	caffe::BlobProto * nextBiasBlob = nextLayer->mutable_blobs(BiasType);
	int nextChannels = nextWightsBlob->channels();
	//transform_blob(nextWightsBlob,std::vector <int>(0),cut_num_list);
	LOG(INFO)<<"next num :"<< nextWightsBlob->num()
	<< "next channels: "<<nextWightsBlob->channels()
	<<"next height: " <<nextWightsBlob->height()
	<<"next width:" <<nextWightsBlob->width();
	//Only for VGG19
	nextWightsBlob->set_num(4096);
	nextWightsBlob->set_channels(512);
	nextWightsBlob->set_height(7);
	nextWightsBlob->set_width(7);
	transform_blob(nextWightsBlob,std::vector<int>(0),cut_num_list);
	nextWightsBlob->set_num(1);
	nextWightsBlob->set_channels(1);
	nextWightsBlob->set_height(4096);
	nextWightsBlob->set_width(7*7*newNum);

	return 0;

}
int cutFilterNumByL1(caffe::LayerParameter * currentLayer,caffe::LayerParameter  * nextLayer,float cutRate) //裁剪当前过滤器的时候同时要改变下一层的通道数
{
	LOG(INFO)<<"Cut current Layer type : "<<currentLayer->type()
		<<"name : "<<currentLayer->type();
	LOG(INFO)<<"Cut next Layer type : "<<nextLayer->name()
		<<"name : "<<nextLayer->name();
	CHECK_EQ(nextLayer->type(),"Convolution");

	caffe::BlobProto * wightsBlob = currentLayer->mutable_blobs(WeightsType);
	caffe::BlobProto * biasBlob = currentLayer->mutable_blobs(BiasType);
	int currentNum = wightsBlob->num();
	int currentChannels = wightsBlob->channels();
	int currentHeight = wightsBlob->height();
	int currentWidth = wightsBlob->width();

	if(!(wightsBlob->has_num() || wightsBlob->has_channels() ||
			wightsBlob->has_height() || wightsBlob->has_width()))
	{
		LOG(INFO) << "use shape .." ;
		currentNum = wightsBlob->shape().dim(0);
		currentChannels = wightsBlob->shape().dim(1);
		currentHeight = wightsBlob->shape().dim(2);
		currentWidth = wightsBlob->shape().dim(3);
	}

	CHECK_LT(cutRate,1.0)<< "cut Radio less than 1.0";
	int newNum = (1-cutRate)*currentNum;

	::google::protobuf::RepeatedField<float> * data = wightsBlob->mutable_data();
	std::vector <std::pair <float,int> > L1_list;
	for(int n = 0;n<currentNum;++n)  //计算每个过滤器的 绝对值之和
	{
		float abs_sum = 0;
		for(int c = 0;c<currentChannels;++c)
		{
			for(int h = 0;h<currentHeight;++h)
			{
				for(int w = 0;w<currentWidth;++w)
				{
					abs_sum += std::abs(data->Get(offset(wightsBlob,n,c,h,w)));
				}
			}
		}
		LOG(INFO) <<"abs_sum of index "<<n<<" : "<<abs_sum;
		L1_list.push_back(std::make_pair(abs_sum,n));
	}
	sort(L1_list.begin(),L1_list.end());  //排序
	std::vector <int> cut_num_list;
	for(int i =0;i<L1_list.size();++i) if(i < currentNum-newNum)
	{
		cut_num_list.push_back(L1_list[i].second);
		LOG(INFO)<<"cut index :"<<L1_list[i].second;
	}

	transform_blob(wightsBlob,cut_num_list); //裁剪操作
	transform_blob(biasBlob,std::vector<int>(0),std::vector<int>(0),std::vector<int>(0),cut_num_list); //bias 裁剪width



	
	if(nextLayer->mutable_blobs()->size() >=2)
	{
		caffe::BlobProto * nextWightsBlob = nextLayer->mutable_blobs(WeightsType);
		caffe::BlobProto * nextBiasBlob = nextLayer->mutable_blobs(BiasType);
		int nextChannels = nextWightsBlob->channels();
		if(! nextWightsBlob->has_channels())
			nextChannels = nextWightsBlob->shape().dim(1);
		CHECK_EQ(currentNum,nextChannels);  //下一层的channel数和当前层的个数相同
		transform_blob(nextWightsBlob,std::vector <int>(0),cut_num_list);
	}
	return 0;
}



//arg 1: wights to cut 
//arg 2: new model name
//arg 3: cut rate (0,1]
//arg 4: cut level [0,conv_level-1]
int main(int argc ,char * argv[])
{

	CHECK_GT(argc,2)<<"at least two args ..";
	LOG(INFO)<<"arg num" << argc;
	NetParameter param;

	ReadNetParamsFromBinaryFileOrDie(argv[1],&param);
	NetParameter originparam(param) ;
	
    param.mutable_state()->set_phase(TEST);
    param.mutable_state()->set_level(0);
	int num_layers = param.layer_size();

	float cut_rate = 0.1f; //默认裁剪率为0.1
	int cut_level = 0; // 默认裁剪第一层
	if(argc >=4)
	{
		cut_rate = atof(argv[3]);
	}
	if(argc >=5)
	{
		cut_level = atoi(argv[4]);
	}

    int last_num = 0;
    int first = 1;
    int current_layer_num = 0;
    int target_layer_num = 0;
	std::vector <caffe::LayerParameter *> layer_list;
	std::vector <caffe::LayerParameter *> inner_product_layer_list;

    for(int i =0;i<num_layers;++i)
    {
        DLOG(ERROR)<<"Layer" <<i<<":" <<param.layer(i).name()<<"\t"
            << param.layer(i).type();
        if(param.layer(i).type() == "Convolution"  )
        {
        	layer_list.push_back(param.mutable_layer(i));
        }
        if(param.layer(i).type() == "InnerProduct")
		{
			inner_product_layer_list.push_back(param.mutable_layer(i));
		}
    }
    //CHECK_LT(cut_level+1,layer_list.size()-1)<< "unable to cut last Convolution layer.. maybe  update later ..";
    if(cut_level+1 == layer_list.size()) // 需要裁剪最后一层
	{
		cutLastLayerByL1(layer_list[cut_level],inner_product_layer_list[0],cut_rate);
	}
	else 
	{
		cutFilterNumByL1(layer_list[cut_level],layer_list[cut_level+1],cut_rate);
	}
	WriteProtoToBinaryFile(param,argv[2]);
    return 0;
}
