
#include "glut.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static unsigned char *pixel_data = NULL;
static int width = 0, height = 0;
static GLuint texture_id = 0;
static char *filename = NULL;

void get_color(unsigned char index, unsigned char *r, unsigned char *g, unsigned char *b) {
    if(index == 0) {
        *r=0; *g=0; *b=0;
        return;
    }

    float t = (float) index / 255.0f;
    *r = (unsigned char)(sin(t * 2.0 * M_PI) * 127 + 128);
    *g = (unsigned char)(sin((t + 1.0/3.0) * 2.0 * M_PI) * 127 + 128);
    *b = (unsigned char)(sin((t + 2.0/3.0) * 2.0 * M_PI) * 127 + 128);
}

int main(int argc, char **argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <datafile.bin>\n", argv[0]);
        return 1;
    }
    filename = argv[1];

    



    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    
}