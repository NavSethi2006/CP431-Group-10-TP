
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

// Color map - map escape index (0..255) to RGB
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

// Build an RGB texture from the escape index data
void build_texture(void) {
    unsigned char *rgb_data = malloc(width * height * 3);
    if(!rgb_data) {
        fprintf(stderr, "Failed to allocate RGB buffer\n");
        exit(1);
    }

    for(int i = 0; i < width * height; i++) {
        unsigned char r, g, b;
        get_color(pixel_data[i], &r, &g, &b);
        rgb_data[i*3 + 0] = r;
        rgb_data[i*3 + 1] = g;
        rgb_data[i*3 + 2] = b;
    }

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_data);
    free(rgb_data);
}

// Read the binary file. Returns 0 on success, 1 on error
int read_data_file(const char *fname) {
    FILE *f = fopen(fname, "rb");
    if (!f) {
        perror("Cannot open file");
        return 1;
    }
    // waiting to see how the data files are formatted

}

// Saves the current opengl window as a jpeg
void save_jpeg(const char *outfile) {
    // Read pixels from the frame buffer
    unsigned char *pixels = malloc(width * height * 3);
    if(pixels) {
        fprintf(stderr, "Out of memory to take a screenshot\n");
        return;
    }
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,pixels);

    // flip verticall because opengl's origin is bottom-left
    unsigned char *flipped = malloc(width * height * 3);
    for(int i = 0; i < height; i++) {
        memcpy(flipped + i * width * 3, pixels + (height - 1 - i)*width*3, width*3);
    }

    // write jpeg
    if(stbi_write_jpg(outfile, width, height, 3, flipped, 90) == 0) {
        fprintf(stderr, "Failed to write jpeg file %s\n", outfile);
    } else {
        printf("Saved %s\n", outfile);
    }

    // free pointers after done
    free(pixels);
    free(flipped);
}

// Display callback
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();

    static int saved = 0;
    if(!saved) {
        char outfile[256];
        snprintf(outfile, sizeof(outfile), "%s.jpg", filename ? filename : "julia");
        save_jpeg(outfile);
        saved = 1;
    }
}

// reshape callback - maintain aspect ratio and set viewport
void reshape(int w, int h) {
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y) {
    if(key=='q' || key == 27) exit(0);
}

int main(int argc, char **argv) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <datafile.bin>\n", argv[0]);
        return 1;
    }
    filename = argv[1];

    if(read_data_file(filename) != 0) {
        return 1;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Julia Set Viewer");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    build_texture();

    glutMainLoop();

    free(pixel_data);
    glDeleteTextures(1, &texture_id);

    return 0;
}