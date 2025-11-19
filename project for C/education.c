#include <stdio.h>  
#include <stdlib.h>  
#include <time.h> 
#include <locale.h>

int main(void)
{
   
    
    int n = 0;
    printf("Please, enter size array to continue: ");
    scanf("%d", &n);
    int array[n];
    
    if (n <= 4)
    {
        printf("Please, enter element array: \n");
        for(int i = 0; i < n; i++)
        {
            printf("Enter %d element array: ", i);  
            scanf("%d", &array[i]);
        }
    }
    else
    {
        srand(time(NULL));
        for(int i = 0; i < n; i++) 
        {
            array[i] = rand() % 201 - 100; 
        }
    }
    
    
    printf("Array: ");
    for(int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    
    
    int divisor;
    printf("Enter divisor: ");
    scanf("%d", &divisor);
    
   
    int findNum = FindNumber(array, n, divisor);
    
    if(findNum != -1000){
        printf("Maximum number not divisible by %d is: %d\n", divisor, findNum);
    }
    else{
        printf("No number found that is not divisible by %d!\n", divisor);
    }

    int findNum2 = FindNumber2(array, n);
    if(findNum2 != -1){
        printf("Index value = %d\n", findNum2);
    }
    else{
        printf("Index valued no found\n");
    }
    return 0;   
}


int FindNumber(int array[], int n, int divisor)
{
    if (divisor == 0) {
        printf("Error: divisor cannot be zero!\n");
        return -1;
    }
    
    int maxNonMultiple = -1000;
    
    for(int i = 0; i < n; i++)
    {
        if (array[i] % divisor != 0) {
            if (maxNonMultiple == -1000 || array[i] > maxNonMultiple) {
                maxNonMultiple = array[i];
            }
        }
    }

    return maxNonMultiple;
}

int FindNumber2(int array[], int n)
{
    int indexMinValue = -1;
    for(int i = 0; i < n; i++)
    {
        if(array[i] < 0)
        {
            if(array[i] % 2 != 0){
                indexMinValue = i;
            }
        }

    }

    return indexMinValue;
}


