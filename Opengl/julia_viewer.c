
 #include <GL/glew.h>
 #include <GL/glut.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 
 static const double C_RE = 0.285;
 static const double C_IM = 0.01;
 
 static double view_cx = 0.0, view_cy = 0.0;
 static double view_scale = 3.0;
 static double zoom_velocity = 0.0;
 static double mouse_nx = 0.5, mouse_ny = 0.5;
 
 static int keys[256] = {0};
 static int win_w = 900, win_h = 900;
 
 static int max_iter = 512;
 
 static GLuint shader_prog = 0;
 static GLuint vao = 0, vbo = 0;
 
 static GLint u_center, u_scale, u_resolution, u_max_iter, u_c;
 
 static const char *VERT_SRC =
     "#version 400 core\n"
     "layout(location=0) in vec2 pos;\n"
     "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
 
 /*
  * Fragment shader: computes one Julia iteration per fragment.
  * Uses smooth colouring (continuous dwell) + a rich palette.
  */
 static const char *FRAG_SRC =
     "#version 400 core\n"
     "#extension GL_ARB_gpu_shader_fp64 : enable\n"
     "out vec4 fragColor;\n"
     "\n"
     "uniform dvec2  u_center;\n"
     "uniform double u_scale;\n"
     "uniform vec2   u_resolution;\n"
     "uniform int    u_max_iter;\n"
     "uniform dvec2  u_c;\n"
     "\n"
     "void main() {\n"
     "    dvec2 uv = dvec2(gl_FragCoord.xy) / dvec2(u_resolution);\n"
     "    double aspect = double(u_resolution.x) / double(u_resolution.y);\n"
     "    dvec2 z = dvec2((uv.x - 0.5) * u_scale * 2.0 * aspect,\n"
     "                    (uv.y - 0.5) * u_scale * 2.0) + u_center;\n"
     "\n"
     "    int i = 0;\n"
     "    for(; i < u_max_iter; i++) {\n"
     "        double x2 = z.x*z.x, y2 = z.y*z.y;\n"
     "        if(x2 + y2 > 256.0) break;\n"
     "        z = dvec2(x2 - y2, 2.0*z.x*z.y) + u_c;\n"
     "    }\n"
     "\n"
     "    if(i == u_max_iter) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
     "\n"
     "    /* smooth colouring — all in float to avoid log() ambiguity */\n"
     "    float zx = float(z.x), zy = float(z.y);\n"
     "    float log_zn = log(zx*zx + zy*zy) * 0.5;\n"
     "    float nu     = log(log_zn / log(2.0)) / log(2.0);\n"
     "    float t      = clamp((float(i) + 1.0 - nu) / float(u_max_iter), 0.0, 1.0);\n"
     "\n"
     "    float r = 0.5 + 0.5*cos(6.28318*(t*3.0 + 0.00));\n"
     "    float g = 0.5 + 0.5*cos(6.28318*(t*3.0 + 0.33));\n"
     "    float b = 0.5 + 0.5*cos(6.28318*(t*3.0 + 0.67));\n"
     "\n"
     "    fragColor = vec4(r, g, b, 1.0);\n"
     "}\n";
 
 static GLuint compile_shader(GLenum type, const char *src) {
     GLuint s = glCreateShader(type);
     glShaderSource(s, 1, &src, NULL);
     glCompileShader(s);
 
     GLint ok;
     glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
     if(!ok) {
         char log[2048];
         glGetShaderInfoLog(s, sizeof(log), NULL, log);
         fprintf(stderr, "Shader compile error:\n%s\n", log);
         exit(1);
     }
     return s;
 }
 
 static void build_shader() {
     GLuint vs = compile_shader(GL_VERTEX_SHADER,   VERT_SRC);
     GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
 
     shader_prog = glCreateProgram();
     glAttachShader(shader_prog, vs);
     glAttachShader(shader_prog, fs);
     glLinkProgram(shader_prog);
 
     GLint ok;
     glGetProgramiv(shader_prog, GL_LINK_STATUS, &ok);
     if(!ok) {
         char log[2048];
         glGetProgramInfoLog(shader_prog, sizeof(log), NULL, log);
         fprintf(stderr, "Program link error:\n%s\n", log);
         exit(1);
     }
 
     glDeleteShader(vs);
     glDeleteShader(fs);
 
     u_center     = glGetUniformLocation(shader_prog, "u_center");
     u_scale      = glGetUniformLocation(shader_prog, "u_scale");
     u_resolution = glGetUniformLocation(shader_prog, "u_resolution");
     u_max_iter   = glGetUniformLocation(shader_prog, "u_max_iter");
     u_c          = glGetUniformLocation(shader_prog, "u_c");
 }
 
 static void build_quad() {
     float verts[] = { -1,-1,  1,-1,  -1,1,  1,1 };
 
     glGenVertexArrays(1, &vao);
     glBindVertexArray(vao);
 
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
 
     glEnableVertexAttribArray(0);
     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
 }
 
 static void load_bin(const char *fname) {
     FILE *f = fopen(fname, "rb");
     if(!f) { perror("open"); exit(1); }
 
     int w, h;
     fread(&w, sizeof(int), 1, f);
     fread(&h, sizeof(int), 1, f);
     fclose(f);
 
     printf("Loaded .bin: %d x %d  (used for dimension info only)\n", w, h);
     printf("Julia constant: c = %.4f + %.4fi\n", C_RE, C_IM);
     printf("Controls: Scroll=zoom  WASD=pan  +/-=quality  R=reset  ESC=quit\n");
 }
 
 static void zoom_toward_mouse(double factor) {
     double aspect = (double)win_w / win_h;
     double fx = (mouse_nx - 0.5) * view_scale * 2.0 * aspect + view_cx;
     double fy = (mouse_ny - 0.5) * view_scale * 2.0         + view_cy;
 
     view_scale *= factor;
 
     view_cx = fx - (mouse_nx - 0.5) * view_scale * 2.0 * aspect;
     view_cy = fy - (mouse_ny - 0.5) * view_scale * 2.0;
 }
 
 static void reset_view() {
     view_cx = 0.0; view_cy = 0.0; view_scale = 1.5;
     zoom_velocity = 0.0;
 }
 
 static void display() {
     glClear(GL_COLOR_BUFFER_BIT);
 
     glUseProgram(shader_prog);
 
     glUniform2d(u_center,     view_cx, view_cy);
     glUniform1d(u_scale,      view_scale);
     glUniform2f(u_resolution, (float)win_w, (float)win_h);
     glUniform1i(u_max_iter,   max_iter);
     glUniform2d(u_c,          C_RE, C_IM);
 
     glBindVertexArray(vao);
     glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
 
     glutSwapBuffers();
 }
 
 static void reshape(int w, int h) {
     win_w = w; win_h = h;
     glViewport(0, 0, w, h);
 }
 
 static void passive_motion(int x, int y) {
     mouse_nx = (double)x / win_w;
     mouse_ny = 1.0 - (double)y / win_h;
 }
 
 static void mouse_btn(int button, int state, int x, int y) {
     if(state != GLUT_DOWN) return;
     passive_motion(x, y);
     if(button == 3) zoom_velocity -= 0.04;   /* scroll up   = zoom in  */
     if(button == 4) zoom_velocity += 0.04;   /* scroll down = zoom out */
 }
 
 static void key_down(unsigned char k, int x, int y) {
     keys[(int)k] = 1;
     if(k == 27) exit(0);
     if(k == 'r' || k == 'R') { reset_view(); glutPostRedisplay(); }
     if(k == '+' || k == '=') { max_iter = (int)(max_iter * 1.5); printf("max_iter = %d\n", max_iter); glutPostRedisplay(); }
     if(k == '-')              { max_iter = max_iter > 64 ? (int)(max_iter / 1.5) : 64; printf("max_iter = %d\n", max_iter); glutPostRedisplay(); }
 }
 
 static void key_up(unsigned char k, int x, int y) { keys[(int)k] = 0; }
 
 static void idle() {
     int changed = 0;
 
     /* WASD pan — speed proportional to current zoom level */
     double pan = view_scale * 0.03;
     if(keys['w'] || keys['W']) { view_cy += pan; changed = 1; }
     if(keys['s'] || keys['S']) { view_cy -= pan; changed = 1; }
     if(keys['a'] || keys['A']) { view_cx -= pan; changed = 1; }
     if(keys['d'] || keys['D']) { view_cx += pan; changed = 1; }
 
     /* smooth zoom with momentum */
     if(fabs(zoom_velocity) > 1e-5) {
         zoom_toward_mouse(1.0 + zoom_velocity);
         zoom_velocity *= 0.88;
         changed = 1;
     }
 
     if(changed) glutPostRedisplay();
 }
 
 int main(int argc, char **argv) {
     if(argc < 2) {
         fprintf(stderr, "Usage: %s julia_set.bin\n", argv[0]);
         return 1;
     }
 
     load_bin(argv[1]);
 
     glutInit(&argc, argv);
     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
     glutInitWindowSize(win_w, win_h);
     glutCreateWindow("Julia Set — Infinite Zoom");
 
     GLenum err = glewInit();
     if(err != GLEW_OK) {
         fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(err));
         return 1;
     }
 
     /* require OpenGL 3.3 for dvec2 uniforms + VAO */
     printf("OpenGL: %s\n", glGetString(GL_VERSION));
     printf("GLSL:   %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
 
     build_shader();
     build_quad();
     reset_view();
 
     glutDisplayFunc(display);
     glutReshapeFunc(reshape);
     glutPassiveMotionFunc(passive_motion);
     glutMotionFunc(passive_motion);
     glutMouseFunc(mouse_btn);
     glutKeyboardFunc(key_down);
     glutKeyboardUpFunc(key_up);
     glutIdleFunc(idle);
 
     glutMainLoop();
     return 0;
 }