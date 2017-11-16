编译方法：
工程由cmake编译。
在当前目录下执行cmake . (“.” 表示当前目录)
会在当前目录下生成makefile文件
然后make 编译生成二进制的.bin文件

注意：CMakeLists file 中的CAFFE_ROOT 路径需要与本机的caffe的目录对应。


二进制文件用法:

 deal_model.bin 接受2到4个参数
第一个参数为要裁剪的二进制caffemodel文件名: 如 VGG_ILSVRC_19_layers.caffemodel
第二个参数为要新生成的二进制caffemodel 文件名
如： VGG_19_conv1_1_cutrate_0.1.caffemodel
第三个参数为裁剪率： 范围从0-1 ，默认为0.1 
第四个参数为裁剪层级： 范围从0-卷积层-1 ,默认为0，表示裁剪第1层。

详情见源码。

测试方法：

1.	首先需要修改网络结构定义文件：
比如我裁剪了第一个的卷积层的绝对值最小的10%的过滤器，我就需要把prototxt文件中的对应的参数修改过来。
以VGG19为例，需要将num_output 从64变成90%，即57. 
 
注意: VGG19原版的prototxt文件有一些错误:
 每层的定义layers 需要改成 layer 。
然后诸如 CONVOLUNTION 这种参数 需要改成用双引号包括的 “Convolution”,



然后当前目录下应该有裁剪过后的caffemodel 文件，以及修改后的网络结构定义文件
如：VGG_19_conv1_1_cutrate_0.1.caffemodel 和 VGG_19_origin.prototxt

使用原版caffe程序进行测试:

$CAFFE_ROOT/build/tools/caffe.bin test \
				-model	VGG_19_origin.prototxt \
				-weights VGG_19_conv1_1_cutrate_0.1.caffemodel \
				-iterations 100 \
				-gpu 0 
参见test_cut.sh


