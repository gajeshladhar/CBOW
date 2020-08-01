#include<iostream>
#include<vector>
#include<fstream>
#include "Eigens/eigen-master/Eigen/Dense"
#include "CBOW.h"

using namespace std;

int main()
{
    string corpus;
    ifstream fin;
    fin.open("data.txt");
    getline(fin,corpus);
    
    fin.close();
    
    CBOW* model=new CBOW(100,4,corpus);
    model->run(1.0,1);

    // model->print_vocab();
    //cout<<endl<<"Piyush : "<<endl<<model->get_embedding("piyush");
    cout<<endl;

    return 0;
}