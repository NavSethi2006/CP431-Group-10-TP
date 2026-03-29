@echo off

gcc julia_viewer.c -lglew32 -lfreeglut -lopengl32 -lglu32 -o julia_viewer.exe
,
if %errorlevel%==0 (
    echo ✅ Build successful!
    echo Run with: viewer.exe
) else (
    echo ❌ Build failed
)