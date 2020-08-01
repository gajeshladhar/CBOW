#include<map>
#include<cmath>

using namespace std;


class CBOW
{
    private :

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> W1;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> b1;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> W2;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> b2;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> X;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Y;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z1;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A1;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z2;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A2;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dZ1;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dW1;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dZ2;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dW2;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> db1;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> db2;

    string corpus;
    vector<string> words;
    map<int,string> reverse_vocab;
    map<string,int> vocab;

    int N;
    int Window;
    double loss;
    double alpha=0.001;
    int total_exps;

    public :

    CBOW(int N,int Window,string corpus)
    {
        
        this->N=N;
        this->Window=Window;
        this->corpus=corpus;
        
        vocab_init();
        this->total_exps=words.size()-2*Window-1;
       
        prepare_XY();
        weights_init();
    }

    private : void vocab_init()
    {
        string temp="";
       for(int i=0;i<corpus.length();i++)
       {
          
           if(corpus[i]==' ')
           {
               
               
               if( temp.length()!=0 && temp!=" ")
               {
                words.push_back(temp);
                if( vocab.find(temp)==vocab.end())
                {
               vocab[temp]=vocab.size();
               reverse_vocab[reverse_vocab.size()]=temp;
                }
               }
               
               temp="";
               continue;
           }
           
           temp+=corpus[i];
           
       }
        
    }

    private : void prepare_XY()
    {

        static int i=0;
        X.resize(100,vocab.size()) ;
        Y.resize(100,vocab.size()) ;

        X.array()=0.0;
        Y.array()=0.0;

        i=i%(total_exps-100);
       
        int k;
        for(k=i;k<i+X.rows();k++)
        {
            for(int j=k;j<k+Window;j++)
            X(k-i,vocab[words[j]])+=1.0/Window;

            for(int j=k+Window+1;j<k+2*Window+1;j++)
            X(k-i,vocab[words[j]])+=1.0/Window;

            Y(k-i,vocab[words[k+Window]])=1.0;
        }
        i=k;
    }

    private : void weights_init()
    {
        W1.resize(N,vocab.size());
        W2.resize(vocab.size(),N);

        b1.resize(N,1);
        b2.resize(vocab.size(),1);
        b1.array()=0.0;
        b2.array()=0.0;
        
        for(int i=0;i<W1.rows();i++)
        {
            for(int j=0;j<W1.cols();j++)
            W1(i,j)=(random()*1.0/(1.0+RAND_MAX))*1e-5;
        }

        for(int i=0;i<W2.rows();i++)
        {
            for(int j=0;j<W2.cols();j++)
            W2(i,j)=(double)(random()*1.0/(1.0+RAND_MAX))*1e-5;
        }

    }

    private : void forward_pass()
    {
        Z1=(X*(W1.transpose()))+((b1.transpose()).replicate(X.rows(),1));
        A1=(1.0/exp(1.0+exp(0-Z1.array()).array()));

        Z2=A1*W2.transpose()+((b2.transpose()).replicate(A1.rows(),1));
        A2=exp(Z2.array())/( (exp(Z2.array()).rowwise().sum()).replicate(1,Z2.cols()) );

        loss=(0-(log(A2.array()).cwiseProduct(Y.array())).sum() )/Y.rows();
    
    }

    private : void backward_pass()
    {
        dZ2=A2-Y;
        dW2=((dZ2.transpose())*A1)/dZ2.rows();
        db2=(dZ2.colwise().sum()).transpose();

        dZ1=(dZ2*W2).array()*(A1.array()*(1-A1.array()).array()).array();
        dW1=((dZ1.transpose())*X)/dZ1.rows();
        db1=(dZ1.colwise().sum()).transpose();
    }

    private : void update_weights()
    {
        W1-=alpha*dW1;
        b1-=alpha*db1;

        W2-=alpha*dW2;
        b2-=alpha*db2;

    }


    public : Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> get_embedding(string word)
    {
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> result;
        result.resize(N,1);
        result.array()=0.0;
        if(vocab.find(word)!=vocab.end())
        {
            result=((W1+W2.transpose())/2)(Eigen::seq(0,N-1),Eigen::seq(vocab[word],vocab[word]));
        }
        return result;
    }

    public : void print_vocab()
    {
        for(auto itr=vocab.begin();itr!=vocab.end();itr++)
        cout<<(itr->first)<< " : "<<itr->second<<endl;
    }

    public : void run(double lr,int epochs)
    {
        this->alpha=lr;
        for(int i=0;i<epochs;i++)
        {
        
        for(int j=0;j<(total_exps/100);j++)
        {
        forward_pass();
        backward_pass();
        update_weights();
        cout<<"\nEpoch "<<(i+1)<<"\nBatch : "<<(j+1)<<"\n"<<"Loss : "<<loss;
        prepare_XY();
        }
        }



    }

    


};