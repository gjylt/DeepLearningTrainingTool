#include <iostream>
#include "tools.h"

using namespace std;
int slidingWindow(int *a,int lenArray,int kernelSize);
int delEA(int *a,int lenArray);
int delShort(int *a,int lenArray, int fps);
int fillSame(int *a,int lenArray);
int forceDel(int *a,int lenArray);
int delErrorBeforePhase(int *a,int lenArray,int phase,int errorPhase);
int delErrorAfterPhase(int *a,int lenArray,int phase,int errorPhase);
int addEA(int *a,int lenArray);
int addLast(int *a,int lenArray);
int phaseConstraint(int *a,int lenArray);

//listUse = ["bg",
//           "Establish access",
//           "Adhesion lysis",
//           "Mobilize the Calot\'s triangle",
//           "Dissect gallbladder from liver bed",
//           "Extract the gallbladder",
//           "Clear the operative region",
//           "clip the cystic artery",
//           "clip the cystic duct",
//           "cut the cystic artery",
//           "cut the cystic duct"
//           ]


/* EA→2 AL→3 MCT→4 DGB→5 EG→1 COR→6 */
/* 1      2   3      4     5    6 */
void runThis(int *arr,int lenArray)
{
    int fps = 1;
    slidingWindow(arr,lenArray,9*fps);
    slidingWindow(arr,lenArray,47*fps);
    delShort(arr,lenArray,fps);
    delEA(arr,lenArray);
    fillSame(arr,lenArray);
    forceDel(arr,lenArray);
    //phaseConstraint(a,lenArray);
    delErrorBeforePhase(arr,lenArray,2,6);
    delErrorBeforePhase(arr,lenArray,3,6);
    delErrorBeforePhase(arr,lenArray,4,6);
    delErrorBeforePhase(arr,lenArray,5,6);
    delErrorBeforePhase(arr,lenArray,2,1);
    delErrorBeforePhase(arr,lenArray,3,1);
    delErrorBeforePhase(arr,lenArray,4,1);
    delErrorBeforePhase(arr,lenArray,5,1);
    delErrorAfterPhase(arr,lenArray,6,4);
    delErrorAfterPhase(arr,lenArray,1,4);
    fillSame(arr,lenArray);
    addEA(arr,lenArray);
//    addLast(arr,lenArray);
}

void LcActioPostProcss(int *arr,int lenArray)
{
    int fps = 1;
    slidingWindow(arr,lenArray,9);
    slidingWindow(arr,lenArray,47);
//    delShort(arr,lenArray,fps);
//    delEA(arr,lenArray);
//    fillSame(arr,lenArray);
//    forceDel(arr,lenArray);
    //phaseConstraint(a,lenArray);
//    delErrorBeforePhase(arr,lenArray,2,6);
//    delErrorBeforePhase(arr,lenArray,3,6);
//    delErrorBeforePhase(arr,lenArray,4,6);
//    delErrorBeforePhase(arr,lenArray,5,6);
//    delErrorBeforePhase(arr,lenArray,2,1);
//    delErrorBeforePhase(arr,lenArray,3,1);
//    delErrorBeforePhase(arr,lenArray,4,1);
//    delErrorBeforePhase(arr,lenArray,5,1);
//    delErrorAfterPhase(arr,lenArray,6,4);
//    delErrorAfterPhase(arr,lenArray,1,4);
//    fillSame(arr,lenArray);
//    addEA(arr,lenArray);
    //addLast(arr,lenArray);
}

/*"Extract the gallbladder",
"Establish access",
"Adhesion lysis",
"Mobilize the Calot's triangle",
"Dissect gallbladder from liver bed",
"Clear the operative region"*/
int slidingWindow(int *a,int lenArray,int kernelSize)
{
    int numClass = 1+6;
    int aCount[numClass];
    int aTemp[lenArray];
    for (int i=0;i<lenArray;i++) aTemp[i]=0;

//    int lenArray = sizeof(a)/sizeof(a[0]);
    for(int i=0;i<=lenArray-kernelSize;i++)
    {
        for(int j=0;j<numClass;j++) aCount[j]=0;
        for(int j=i;j<i+kernelSize;j++)
        {
            aCount[a[j]]+=1;
        }
//        int indexMax = distance(aCount, max_element(aCount, aCount + sizeof(aCount)/sizeof(aCount[0])));
        int indexMax=0;
        for(int j=0;j<numClass;j++)
        {
            if(aCount[j]>indexMax) indexMax=j;
        }
        if (aCount[indexMax]>=kernelSize/2+1 and a[i+kernelSize/2]!=indexMax and indexMax!=0)
        {
            aTemp[i+kernelSize/2]=indexMax;
        }
        else
        {
            aTemp[i+kernelSize/2]=a[i+kernelSize/2];
        }
    }

    for (int i=0;i<lenArray;i++) a[i]=*(aTemp+i);
    return 0;
}

int delEA(int *a,int lenArray)
{
    bool change=0;
    for(int i=0;i<lenArray;i++)
    {
        if(a[i]!=0 and a[i]!=2) change=1;
        if(change==1 and a[i]==2) a[i]=0;
    }
    return 0;
}
int delShort(int *a,int lenArray, int fps)
{
    int param[]={4*fps, 16*fps, 19*fps, 22*fps, 25*fps, 28*fps};
    int phaseIdLast = -1;
    int indexStart = -1;
    int countS = 0;
    bool isin=1;

    for(int i=0;i<lenArray;i++)
    {
        isin=0;
        if(phaseIdLast!=a[i])
        {
            if(phaseIdLast==0 or phaseIdLast==-1)
            {
                phaseIdLast=a[i];
                indexStart=i;
                countS=1;
            }
            else
            {
//                if(param[phaseIdLast-1]>=countS and ((a[indexStart]==2 and isin) or a[indexStart]!=2))
                if(param[phaseIdLast-1]>=countS)
                {
                    for(int j=indexStart;j<i;j++) a[j]=0;
                    phaseIdLast=a[i];
                    indexStart=i;
                    countS=1;
                }
                else
                {
                    phaseIdLast=a[i];
                    indexStart=i;
                    countS=1;
                }
            }
        }
        else
        {
            countS+=1;
        }
    }
    if (phaseIdLast==a[lenArray-1] and param[phaseIdLast-1]>=countS and a[lenArray-1]!=2)
    {
        for(int j=indexStart;j<lenArray;j++) a[j]=0;
    }
    return 0;
}
int fillSame(int *a,int lenArray)
{
    int phaseIdLast = -1;
    int indexStart = -1;

    for(int i=0;i<lenArray;i++)
    {
        if(i!=0 and a[i-1]!=0 and a[i]==0)
        {
            phaseIdLast=a[i-1];
            indexStart=i;
        }
        if(a[i]!=0 and a[i-1] ==0 and phaseIdLast==a[i])
        {
            for(int j=indexStart;j<i;j++) a[j]=phaseIdLast;
        }
    }
    return 0;
}

int forceDel(int *a,int lenArray)
{
    float ratio[]={0.6,0.3};
    float ratioC;
    int b[lenArray];
    for(int i=0;i<lenArray;i++) b[i]=a[i];

    int classes[lenArray];
    int startInd[lenArray];
    int endInd[lenArray];
    int indexChange = 0;
    int phaseIdLast=-1;
    int indexNo0;

    indexNo0=lenArray;
    for(int i=0;i<lenArray;i++) classes[i]=0;
    for(int i=0;i<lenArray;i++) startInd[i]=0;
    for(int i=0;i<lenArray;i++) endInd[i]=0;

    for(int i=0;i<lenArray;i++)
    {
        if(a[i]!=0 and a[i]==phaseIdLast)
        {
            endInd[indexChange-1]=i+1;
        }
        else if(a[i]!=phaseIdLast and a[i]!=0)
        {
            classes[indexChange]=a[i];
            startInd[indexChange]=i;
            endInd[indexChange]=i+1;
        }
        if(a[i]!=0 and a[i]!=phaseIdLast)
        {
            phaseIdLast=a[i];
            indexChange+=1;
        }

    }


    for(int i=0;i<lenArray;i++)
    {
        if(i+2<lenArray and classes[i+2]!=0)
        {
            if (a[startInd[i+1]]==1 or a[startInd[i+1]]==3)
            {
                ratioC=ratio[1];
            }
            else
            {
                ratioC=ratio[0];
            }

//            if(classes[i]==classes[i+2] and ((endInd[i]-startInd[i])*ratioC>(endInd[i+1]-startInd[i+1]) and (endInd[i+2]-startInd[i+2])*ratioC>(endInd[i+1]-startInd[i+1])))
            if((endInd[i]-startInd[i])*ratioC>(endInd[i+1]-startInd[i+1]) and (endInd[i+2]-startInd[i+2])*ratioC>(endInd[i+1]-startInd[i+1]))
            {
                for(int j=startInd[i+1];j<endInd[i+1];j++) a[j]=0;
            }
        }
    }
    return 0;
}

int delErrorBeforePhase(int *a,int lenArray,int phase,int errorPhase)
{
    int indPhase=-1;
    for(int i=0;i<lenArray;i++)
    {
        if(a[i]==phase)
        {
            indPhase=i;
            break;
        }
    }
    if(indPhase!=-1)
    {
        for(int i=0;i<indPhase;i++)
        {
            if (a[i]==errorPhase)
            {
                a[i]=0;
            }
        }
    }
    return 0;
}

int delErrorAfterPhase(int *a,int lenArray,int phase,int errorPhase)
{
    bool change=0;

    for(int i=0;i<lenArray;i++)
    {
        if(a[i]==phase)
        {
            change=1;
        }
        if(change and a[i]==errorPhase) a[i]=0;
    }
    return 0;
}

int addEA(int *a,int lenArray)
{
    bool change=1;
    int sumArray = 0;
    for(int i=0;i<lenArray;i++) sumArray+=a[i];

    for(int i=0;i<lenArray;i++)
    {
        if(a[i]!=0) change=0;
        if(change and a[i]==0 and sumArray!=0) a[i]=2;
    }
    return 0;
}

int addLast(int *a,int lenArray)
{
    for(int i=1;i<lenArray;i++)
    {
        if(a[i]==0 and a[i-1]!=0) a[i]=a[i-1];
    }
    return 0;
}

/* EA→2 AL→3 MCT→4 DGB→5 EG→1 COR→6 */
int phaseConstraint(int *a,int lenArray)
{
    int phaseLase=0;
    bool isJump=true;

    for (int i=0;i<lenArray-1;i++)
    {
        /* 开始不可能是EG or COR */
        /*if ((a[i]==1 || a[i]==6) && isJump)*/
        if ((a[i]!=2) && isJump)
        {
            a[i]=0;
        }
        else if(a[i]==2)
        {
            isJump=false;
        }

        for (int j=i;j>=0;j--)
        {
            if (a[j]!=0)
            {
                phaseLase=a[j];
                break;
            }
        }
        switch (phaseLase)
        {
        case 0:
            if (a[i+1]==1 || a[i+1]==6)
            {
                a[i+1]=0;
            }
            break;
        case 2:
            if (a[i+1]==1 || a[i+1]==6)
            {
                a[i+1]=0;
            }
            break;
        case 3:
            if (a[i+1]==2 || a[i+1]==6)
//            if (a[i+1]==1 || a[i+1]==2 || a[i+1]==6)
            {
                a[i+1]=0;
            }
            break;
        case 4:
            if (a[i+1]==2)
            {
                a[i+1]=0;
            }
            break;
        case 5:
            if (a[i+1]==2)
            {
                a[i+1]=0;
            }
            break;
        case 1:
            if (a[i+1]==2)
            {
                a[i+1]=0;
            }
            break;
        case 6:
            if (a[i+1]==2)
            {
                a[i+1]=0;
            }
            break;
        }
    }
    return 0;
}
