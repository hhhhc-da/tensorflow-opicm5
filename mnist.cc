#include <opencv2/opencv.hpp>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/graph.pb.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <random>
#include <chrono>

#include <cstdlib>
#include <cstring>
#include <cstdio>

int main(void){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> randint(0, 9);

	int random_id = randint(gen);
	std::cout << "(process) Random intteger: " << random_id << std::endl;

	const std::string export_dir = "model";

	std::ostringstream oss;
	oss << "picture/" << random_id << ".jpg";
	const char* picture_name = oss.str().c_str();
	std::ifstream ifs("picture/label.txt");
	std::string labels;

	// Get labels
	std::getline(ifs, labels);
	int label = *(labels.begin() + random_id) - '0';
	std::cout << "(process) Load label: " << label << std::endl;
	
	tensorflow::SavedModelBundle bundle;
	tensorflow::SessionOptions session_options;
	tensorflow::RunOptions run_options;
	std::unordered_set<std::string> tags = { tensorflow::kSavedModelTagServe };

	// Load model
	tensorflow::LoadSavedModel(session_options, run_options, export_dir, tags, &bundle);
	// Graph
	std::cout << "(tensorflow) Graph nodes." << std::endl;
	const tensorflow::GraphDef& graph = bundle.meta_graph_def.graph_def();
	for (auto& node : graph.node()){
		std::cout << "(tensorflow) Node name: " << node.name() <<std::endl;
	}
	std::cout << "(tensorflow) Graph print done." << std::endl;

	// Create tensor
	tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 28, 28, 1}));
	
	cv::Mat img = cv::imread(picture_name, cv::IMREAD_GRAYSCALE);
        if (img.empty()){
		std::cout << "(opencv) Cannot load images." << std::endl;
	        return -1;
	}
	else {
		std::cout << "(opencv) Loaded image size: ( row: " << img.rows << ", col: " << img.cols << " )" << std::endl;
	}

	img.convertTo(img, CV_32F);
	std::cout << "(opencv) Convert done." << std::endl;

	// Input tensor
	memcpy(input.flat<float>().data(), img.data, img.total() * sizeof(float));
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs{{"serving_default_conv2d_2_input", input}};
	// Output tensor
	std::vector<tensorflow::Tensor> outputs;
	TF_CHECK_OK(bundle.session->Run(inputs, {"StatefulPartitionedCall"}, {}, &outputs));

	auto output_tensor = outputs[0].matrix<float>();
	std::cout << "(tensorflow) Output tensor data: [ " << outputs[0].matrix<float>() << " ]" << std::endl;
	std::cout << "(tensorflow) Original label: " << label << std::endl;
	return 0;
}

