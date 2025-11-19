#include <stdio.h>  
#include <stdlib.h>  
#include <time.h> 
#include <locale.h>
#include <ctype.h>

int main(void)
{
    char * message = "ht4rh45hh56h56trhg65\n hola amigo"; 
    char * filename = "text.txt";

    FILE *fp = fopen(filename, "W");
    if(fp)
    {
        fputs(message, fp);
        fclose(fp);
        printf("File has been written\n");
    }

}
