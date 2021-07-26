#include "tools.h"
#include <iostream>
#include <time.h>

using namespace std;
int main()
{
    int input[] = {};
    int lenArray = sizeof(input) / sizeof(input[0]);
    int *phase;

    // cout << "before in func" << endl;
    // for (int j = 0; j < sizeof(input) / sizeof(input[0]); j++)
    //     cout << input[j] << " ";
    // cout << "" << endl;

    // cout << "after in func" << endl;

    //    for (int i=0;i<lenArray;i++)
    //    {
    //        phase = runThis(input,i);
    //        cout<<phase[i-1]<<" ";
    //    }
    //    cout<<endl;

    phase = runThis(input, lenArray);
    // for (int j = 0; j < lenArray; j++)
    //     cout << phase[j] << " ";
    // cout << "" << endl;
}
