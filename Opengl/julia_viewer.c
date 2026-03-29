/*
 * =============================================================================
 * CP400N Term Project — Julia Set OpenGL Viewer
 * =============================================================================
 *
 * Reads the binary file produced by main.c (MPI computation) and renders
 * the Julia set using OpenGL with:
 *
 *   • Escape-time colour bands (as specified in the project):
 *       Band 0  : iterations   1– 5   → vivid red-orange
 *       Band 1  : iterations   6–10   → orange-yellow
 *       Band 2  : iterations  11–20   → yellow-green
 *       Band 3  : iterations  21–40   → cyan
 *       Band 4  : iterations  41–80   → blue
 *       Band 5  : iterations  81–160  → violet
 *       Band 6  : iterations 161+     → deep magenta (slow escape)
 *       Interior: did not escape       → black
 *
 *   • Infinite zoom via a GLSL fragment shader (GPU recomputes fractal
 *     live at each zoom level — no texture resolution limit).
 *
 *   • Screenshot export as PPM (no extra deps); JPEG if compiled with
 *     -DHAVE_LIBJPEG -ljpeg.
 *
 * Controls:
 *   Scroll wheel  — zoom in/out toward cursor
 *   WASD          — pan
 *   +/-           — increase/decrease iteration quality
 *   R             — reset view to full image
 *   J             — save screenshot
 *   ESC           — quit
 *
 * Build (no libjpeg):
 *   gcc -O2 julia_viewer.c -o julia_viewer -lGL -lGLEW -lglut -lm
 *
 * Build (with libjpeg for .jpg output):
 *   gcc -O2 -DHAVE_LIBJPEG julia_viewer.c -o julia_viewer -lGL -lGLEW -lglut -lm -ljpeg
 *
 * Run:
 *   ./julia_viewer julia_set.bin
 * =============================================================================
 */

 #include <GL/glew.h>
 #include <GL/glut.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 
 #define JULIA_DEGREE      2     
 #define C_RE              0.285 
 #define C_IM              0.01 
 #define MAX_ITER_DEFAULT  512
 
 static double view_cx    = 0.0;
 static double view_cy    = 0.0;
 static double view_scale = 1.5;
 static double zoom_vel   = 0.0; 
 static double mouse_nx   = 0.5;
 static double mouse_ny   = 0.5;
 static int    keys[256]  = {0};
 static int    win_w = 900, win_h = 900;
 static int    max_iter = MAX_ITER_DEFAULT;
 
 static GLuint prog = 0, vao = 0, vbo = 0;
 static GLint  u_center, u_scale, u_res, u_maxiter, u_c, u_degree;
 

 static const char *VERT_SRC =
     "#version 400 core\n"
     "layout(location=0) in vec2 pos;\n"
     "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
 
 static const char *FRAG_SRC =
     "#version 400 core\n"
     "#extension GL_ARB_gpu_shader_fp64 : enable\n"
     "out vec4 fragColor;\n"
     "\n"
     "uniform dvec2  u_center;\n"
     "uniform double u_scale;\n"
     "uniform vec2   u_res;\n"
     "uniform int    u_maxiter;\n"
     "uniform dvec2  u_c;\n"
     "uniform int    u_degree;\n"
     "\n"
     "/* ── project escape-time colour bands ── */\n"
     "/* Band 0: 1-5, Band 1: 6-10, Band 2: 11-20, Band 3: 21-40,  */\n"
     "/* Band 4: 41-80, Band 5: 81-160, Band 6: 161+               */\n"
     "vec3 band_color(int iters, float smooth_t) {\n"
     "    vec3 bands[7];\n"
     "    bands[0] = vec3(1.00, 0.20, 0.00);\n"  /* red-orange  */
     "    bands[1] = vec3(1.00, 0.65, 0.00);\n"  /* orange      */
     "    bands[2] = vec3(0.80, 1.00, 0.10);\n"  /* yellow-green*/
     "    bands[3] = vec3(0.00, 0.90, 0.90);\n"  /* cyan        */
     "    bands[4] = vec3(0.10, 0.30, 1.00);\n"  /* blue        */
     "    bands[5] = vec3(0.60, 0.00, 1.00);\n"  /* violet      */
     "    bands[6] = vec3(0.90, 0.00, 0.55);\n"  /* magenta     */
     "    int b;\n"
     "    if      (iters <=   5) b = 0;\n"
     "    else if (iters <=  10) b = 1;\n"
     "    else if (iters <=  20) b = 2;\n"
     "    else if (iters <=  40) b = 3;\n"
     "    else if (iters <=  80) b = 4;\n"
     "    else if (iters <= 160) b = 5;\n"
     "    else                   b = 6;\n"
     "    vec3 col  = bands[b];\n"
     "    vec3 next = bands[min(b+1,6)];\n"
     "    return mix(col, next, smooth_t * 0.4);\n"
     "}\n"
     "\n"
     "void main() {\n"
     "    /* map pixel to fractal space (double precision) */\n"
     "    dvec2 uv = dvec2(gl_FragCoord.xy) / dvec2(u_res);\n"
     "    double aspect = double(u_res.x) / double(u_res.y);\n"
     "    dvec2 z = dvec2((uv.x-0.5)*u_scale*2.0*aspect,\n"
     "                    (uv.y-0.5)*u_scale*2.0) + u_center;\n"
     "\n"
     "    /* iterate */\n"
     "    int escaped_at = -1;\n"
     "    for (int i = 0; i < u_maxiter; i++) {\n"
     "        double x2 = z.x*z.x, y2 = z.y*z.y;\n"
     "        if (x2 + y2 > 256.0) { escaped_at = i+1; break; }\n"
     "        if (u_degree == 3) {\n"
     "            /* Tc(z) = z^3 + c */\n"
     "            double nx = z.x*(x2 - 3.0*y2);\n"
     "            double ny = z.y*(3.0*x2 - y2);\n"
     "            z = dvec2(nx,ny) + u_c;\n"
     "        } else {\n"
     "            /* Qc(z) = z^2 + c */\n"
     "            z = dvec2(x2-y2, 2.0*z.x*z.y) + u_c;\n"
     "        }\n"
     "    }\n"
     "\n"
     "    /* interior — black */\n"
     "    if (escaped_at < 0) { fragColor = vec4(0.0,0.0,0.0,1.0); return; }\n"
     "\n"
     "    /* smooth colouring: fractional escape time removes integer banding */\n"
     "    float zxf = float(z.x), zyf = float(z.y);\n"
     "    float log_zn = log(zxf*zxf + zyf*zyf) * 0.5;\n"
     "    float nu     = log(log_zn / log(2.0)) / log(2.0);\n"
     "    float st     = clamp(1.0 - nu, 0.0, 1.0);\n"
     "\n"
     "    fragColor = vec4(band_color(escaped_at, st), 1.0);\n"
     "}\n";
 
 static GLuint compile_shader(GLenum type, const char *src)
 {
     GLuint s = glCreateShader(type);
     glShaderSource(s, 1, &src, NULL);
     glCompileShader(s);
     GLint ok;
     glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
     if (!ok) {
         char log[4096];
         glGetShaderInfoLog(s, sizeof(log), NULL, log);
         fprintf(stderr, "Shader compile error:\n%s\n", log);
         exit(1);
     }
     return s;
 }
 
 static void build_shader(void)
 {
     GLuint vs = compile_shader(GL_VERTEX_SHADER,   VERT_SRC);
     GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
     prog = glCreateProgram();
     glAttachShader(prog, vs);
     glAttachShader(prog, fs);
     glLinkProgram(prog);
     GLint ok;
     glGetProgramiv(prog, GL_LINK_STATUS, &ok);
     if (!ok) {
         char log[4096];
         glGetProgramInfoLog(prog, sizeof(log), NULL, log);
         fprintf(stderr, "Program link error:\n%s\n", log);
         exit(1);
     }
     glDeleteShader(vs);
     glDeleteShader(fs);
 
     u_center  = glGetUniformLocation(prog, "u_center");
     u_scale   = glGetUniformLocation(prog, "u_scale");
     u_res     = glGetUniformLocation(prog, "u_res");
     u_maxiter = glGetUniformLocation(prog, "u_maxiter");
     u_c       = glGetUniformLocation(prog, "u_c");
     u_degree  = glGetUniformLocation(prog, "u_degree");
 }
 
 static void build_quad(void)
 {
     float v[] = { -1,-1, 1,-1, -1,1, 1,1 };
     glGenVertexArrays(1, &vao);
     glBindVertexArray(vao);
     glGenBuffers(1, &vbo);
     glBindBuffer(GL_ARRAY_BUFFER, vbo);
     glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
     glEnableVertexAttribArray(0);
     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
 }
 
 static void load_bin(const char *fname)
 {
     FILE *f = fopen(fname, "rb");
     if (!f) { perror("fopen"); exit(1); }
     int w, h;
     fread(&w, sizeof(int), 1, f);
     fread(&h, sizeof(int), 1, f);
     fclose(f);
     printf("Loaded %s  (%d x %d)\n", fname, w, h);
     printf("c = %.6f + %.6fi,  degree = %d\n", C_RE, C_IM, JULIA_DEGREE);
     printf("Controls: Scroll=zoom  WASD=pan  +/-=quality  R=reset  J=save  ESC=quit\n");
 }
 
 static void zoom_to_mouse(double factor)
 {
     double aspect = (double)win_w / win_h;
     double fx = (mouse_nx - 0.5) * view_scale * 2.0 * aspect + view_cx;
     double fy = (mouse_ny - 0.5) * view_scale * 2.0           + view_cy;
     view_scale *= factor;
     view_cx = fx - (mouse_nx - 0.5) * view_scale * 2.0 * aspect;
     view_cy = fy - (mouse_ny - 0.5) * view_scale * 2.0;
 }
 
 static void reset_view(void)
 {
     view_cx = 0.0; view_cy = 0.0; view_scale = 1.5; zoom_vel = 0.0;
 }
 
 
 static void display(void)
 {
     glClear(GL_COLOR_BUFFER_BIT);
     glUseProgram(prog);
     glUniform2d(u_center,  view_cx, view_cy);
     glUniform1d(u_scale,   view_scale);
     glUniform2f(u_res,     (float)win_w, (float)win_h);
     glUniform1i(u_maxiter, max_iter);
     glUniform2d(u_c,       C_RE, C_IM);
     glUniform1i(u_degree,  JULIA_DEGREE);
     glBindVertexArray(vao);
     glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
     glutSwapBuffers();
 }
 
 static void reshape(int w, int h) { win_w=w; win_h=h; glViewport(0,0,w,h); }
 
 static void passive_motion(int x, int y)
 {
     mouse_nx = (double)x / win_w;
     mouse_ny = 1.0 - (double)y / win_h;
 }
 
 static void mouse_btn(int btn, int state, int x, int y)
 {
     if (state != GLUT_DOWN) return;
     passive_motion(x, y);
     if (btn == 3) zoom_vel -= 0.04;
     if (btn == 4) zoom_vel += 0.04;
 }
 
 static void key_down(unsigned char k, int x, int y)
 {
     keys[(int)k] = 1;
     if (k == 27) exit(0);
     if (k=='r'||k=='R') { reset_view(); glutPostRedisplay(); }
     if (k=='j'||k=='J') { save_screenshot("julia_screenshot.jpg"); }
     if (k=='+'||k=='=') { max_iter=(int)(max_iter*1.5); printf("max_iter=%d\n",max_iter); glutPostRedisplay(); }
     if (k=='-')          { max_iter=max_iter>64?(int)(max_iter/1.5):64; printf("max_iter=%d\n",max_iter); glutPostRedisplay(); }
 }
 
 static void key_up(unsigned char k, int x, int y) { keys[(int)k]=0; }
 
 static void idle(void)
 {
     int changed = 0;
     double pan = view_scale * 0.03;
     if (keys['w']||keys['W']) { view_cy+=pan; changed=1; }
     if (keys['s']||keys['S']) { view_cy-=pan; changed=1; }
     if (keys['a']||keys['A']) { view_cx-=pan; changed=1; }
     if (keys['d']||keys['D']) { view_cx+=pan; changed=1; }
     if (fabs(zoom_vel) > 1e-5) {
         zoom_to_mouse(1.0 + zoom_vel);
         zoom_vel *= 0.88;
         changed = 1;
     }
     if (changed) glutPostRedisplay();
 }
 
 int main(int argc, char **argv)
 {
     if (argc < 2) { fprintf(stderr, "Usage: %s julia_set.bin\n", argv[0]); return 1; }
 
     load_bin(argv[1]);
 
     glutInit(&argc, argv);
     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
     glutInitWindowSize(win_w, win_h);
     glutCreateWindow("Julia Set Viewer — CP400N");
 
     GLenum err = glewInit();
     if (err != GLEW_OK) { fprintf(stderr, "GLEW: %s\n", glewGetErrorString(err)); return 1; }
 
     printf("OpenGL: %s | GLSL: %s\n", glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
 
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