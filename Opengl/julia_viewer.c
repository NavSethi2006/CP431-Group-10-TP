#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

GLuint program;

double min_re = -1.5, max_re = 1.5;
double min_im = -1.5, max_im = 1.5;

double zoom_velocity = 0.0;
double mouse_x = 0.5, mouse_y = 0.5;

int keys[256] = {0};

// ===== SHADERS =====
const char* vertex_shader_src =
"#version 120\n"
"void main() { gl_Position = gl_Vertex; }";

const char* fragment_shader_src =
"#version 120\n"
"uniform vec2 minC;\n"
"uniform vec2 maxC;\n"
"uniform vec2 resolution;\n"
"uniform vec2 c;\n"
"\n"
"void main() {\n"
"    vec2 uv = gl_FragCoord.xy / resolution;\n"
"    float x = mix(minC.x, maxC.x, uv.x);\n"
"    float y = mix(minC.y, maxC.y, uv.y);\n"
"\n"
"    vec2 z = vec2(x,y);\n"
"\n"
"    int i;\n"
"    int max_iter = 500;\n"
"\n"
"    for(i = 0; i < max_iter; i++) {\n"
"        if(dot(z,z) > 4.0) break;\n"
"        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;\n"
"    }\n"
"\n"
"    float t = float(i)/float(max_iter);\n"
"\n"
"    vec3 col = vec3(\n"
"        pow(t,0.3),\n"
"        pow(t,0.5),\n"
"        pow(t,0.8)\n"
"    );\n"
"\n"
"    gl_FragColor = vec4(col,1.0);\n"
"}";

// ===== SHADER UTILS =====
GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

void init_shader() {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
}

// ===== RENDER =====
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);

    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);

    glUniform2f(glGetUniformLocation(program,"minC"), min_re, min_im);
    glUniform2f(glGetUniformLocation(program,"maxC"), max_re, max_im);
    glUniform2f(glGetUniformLocation(program,"resolution"), w, h);
    glUniform2f(glGetUniformLocation(program,"c"), -0.7, 0.27015);

    glBegin(GL_QUADS);
    glVertex2f(-1,-1);
    glVertex2f( 1,-1);
    glVertex2f( 1, 1);
    glVertex2f(-1, 1);
    glEnd();

    glutSwapBuffers();
}

// ===== CONTROLS =====
void zoom_to_mouse(double factor) {
    double cx = min_re + mouse_x * (max_re - min_re);
    double cy = min_im + mouse_y * (max_im - min_im);

    double width = (max_re - min_re) * factor;
    double height = (max_im - min_im) * factor;

    min_re = cx - width * mouse_x;
    max_re = min_re + width;

    min_im = cy - height * mouse_y;
    max_im = min_im + height;
}

void motion(int x, int y) {
    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);

    mouse_x = (double)x / w;
    mouse_y = 1.0 - (double)y / h;
}

void mouse(int button, int state, int x, int y) {
    if(state != GLUT_DOWN) return;

    if(button == 3) zoom_velocity -= 0.05;
    if(button == 4) zoom_velocity += 0.05;
}

void key_down(unsigned char key, int x, int y) {
    keys[key] = 1;
    if(key == 27) exit(0);
}

void key_up(unsigned char key, int x, int y) {
    keys[key] = 0;
}

void idle() {
    int changed = 0;

    // WASD pan
    if(keys['w']) { min_im += 0.02; max_im += 0.02; changed=1; }
    if(keys['s']) { min_im -= 0.02; max_im -= 0.02; changed=1; }
    if(keys['a']) { min_re -= 0.02; max_re -= 0.02; changed=1; }
    if(keys['d']) { min_re += 0.02; max_re += 0.02; changed=1; }

    // smooth zoom
    if(fabs(zoom_velocity) > 0.001) {
        zoom_to_mouse(1.0 + zoom_velocity);
        zoom_velocity *= 0.85;
        changed = 1;
    }

    if(changed) glutPostRedisplay();
}

// ===== MAIN =====
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800,800);
    glutCreateWindow("Julia Explorer");

    glewInit();

    init_shader();

    glutDisplayFunc(display);
    glutPassiveMotionFunc(motion);
    glutMouseFunc(mouse);
    glutKeyboardFunc(key_down);
    glutKeyboardUpFunc(key_up);
    glutIdleFunc(idle);

    glutMainLoop();
    return 0;
}