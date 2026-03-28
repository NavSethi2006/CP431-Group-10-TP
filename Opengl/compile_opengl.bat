@echo off

echo Detecting Windows environment...

REM Assume MinGW is installed
set GCC=gcc

REM Paths to bundled libs
set INCLUDE=libs\windows\include
set LIB=libs\windows\lib

echo Compiling...

%GCC% julia_viewer.c -I%INCLUDE% -L%LIB% -lglew32 -lglut32 -lopengl32 -lglu32 -o viewer.exe

if %errorlevel%==0 (
    echo Build complete!
    echo Run with: viewer.exe
) else (
    echo Compilation failed.
)