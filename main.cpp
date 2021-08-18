#include <glog/logging.h>
#include <ceres/ceres.h>

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <iostream>

void saveData(std::vector<std::vector<float>> obs, double* parameters, const int n_params, int lowB, int upperB){
    std::vector<std::vector<float>> curve;
    int step_obs = obs[1][0] - obs[0][0];
    float step = int(step_obs) / 2.;
    for(float sample=lowB; sample<upperB + step; sample+=step){
        std::vector<float> curve_eval;
        double tmp = 0.0;
        for(int i=0; i<n_params; i++){
            tmp += parameters[i]* pow(sample, i);
        }
        curve_eval.push_back(sample);
        curve_eval.push_back(tmp);
        curve.push_back(curve_eval);
    }
    std::ofstream outFile("../results.txt");
    for (int i=0; i<obs.size(); i++){
        outFile << i << "," << obs[i][0] << "," << obs[i][1]<<std::endl;
    }
    for (int i=0; i<curve.size(); i++){
        outFile << i << "," << curve[i][0] << "," << curve[i][1]<<std::endl;
    }
}

std::vector<std::vector<float>> generateMockData(int lowerBound, int upperBound, int step){
    std::vector<std::vector<float>> data;
    std::vector<float> tmp;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(6, 60); // play with values to add gaussian noise

    for(auto i=lowerBound; i<upperBound + step; i+=step){
        tmp.clear();
        tmp.push_back(i);
        tmp.push_back(i*i + dist(generator));
        data.push_back(tmp);
    }
    return data;
}

struct CostFunctor {
public:
    CostFunctor(double x, double y, int degree) : x_(x), y_(y), degree_(degree) {}
    template <typename T>
    // Dynamic autodiff operator signature. Note difference below with autodiff signature
    bool operator()(T const* const* params, T* residual) const {
        // We added only one block of size <degree>. Therefore, all the parameters are in the first memory position
        // pointed by params: params[0]
        T tmp(0.0); // Need a T variable to add the evaluation of the curve
        for (int i=0; i<degree_; i++){
            tmp += params[0][i]*pow(x_, i);
        }
        residual[0] = y_ - tmp;
        return true;
    }

    // Autodiff operator signature
//    bool operator()(T const* params, T* residual) const {
//        T tmp(0);
//        for (int i=0; i<degree_; i++){
//            tmp += params[i]*pow(x_, i);
//        }
//        residual[0] = y_ - tmp;
//        return true;
//    }

private:
    const double x_;
    const double y_;
    const int degree_;
};

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    int lowerBound = -20;
    int upperBound = 20;
    std::vector<std::vector <float>> obs = generateMockData(lowerBound, upperBound, 2);

    //Initialize curve of n coefficients.
    int degree = std::stoi(argv[1]);
    double params[degree]; // the length of the coefficients - 1 indicates the degree of the curve
    memset(params, 0.0, degree * sizeof(float));

    ceres::Problem problem;
    for(auto & ob : obs){
        // We use Dynamic autodiff because the size of the parameters (command line argument) is not known at compile
        // time but at runtime.
        ceres::DynamicAutoDiffCostFunction<CostFunctor, 4>* costFunction =
                new ceres::DynamicAutoDiffCostFunction<CostFunctor>(new CostFunctor(ob[0], ob[1], degree));
        // Fit the curve, i.e., add a block of <degree> parameters per observation.
        costFunction->AddParameterBlock(degree);
        costFunction->SetNumResiduals(1);
        problem.AddResidualBlock(costFunction, nullptr, params);

        // This is used if sizes of parameters were known at compile time
        //        ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<CostFunctor, 1, degree>(new CostFunctor(ob[0], ob[1], degree));
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::cout<<"Initial: ";
    for(int i=0; i<degree; i++){
        std::cout << "0.0, ";
    }
    std::cout<<std::endl;
    std::cout<<"Initial: ";
    for(int i=0; i<degree; i++){
        std::cout << params[i] <<", ";
    }
    std::cout<<std::endl;

    saveData(obs, params, degree, lowerBound, upperBound);

    return 0;

}
