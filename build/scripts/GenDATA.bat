@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      https://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem
@rem  GenDATA startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%"=="" set DIRNAME=.
@rem This is normally unused
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%..

@rem Resolve any "." and ".." in APP_HOME to make it shorter.
for %%i in ("%APP_HOME%") do set APP_HOME=%%~fi

@rem Add default JVM options here. You can also use JAVA_OPTS and GEN_DATA_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS=

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if %ERRORLEVEL% equ 0 goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto execute

echo. 1>&2
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
echo. 1>&2
echo Please set the JAVA_HOME variable in your environment to match the 1>&2
echo location of your Java installation. 1>&2

goto fail

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\lib\GenDATA.jar;%APP_HOME%\lib\polyglot.jar;%APP_HOME%\lib\java_cup.jar;%APP_HOME%\lib\jflex.jar;%APP_HOME%\lib\ppg.jar;%APP_HOME%\lib\pth.jar;%APP_HOME%\lib\efg.jar;%APP_HOME%\lib\checker-qual.jar;%APP_HOME%\lib\plume-util.jar;%APP_HOME%\lib\checker.jar;%APP_HOME%\lib\checker-util.jar;%APP_HOME%\lib\javaparser-core-3.26.2.jar;%APP_HOME%\lib\javac.jar;%APP_HOME%\lib\checker-source.jar;%APP_HOME%\lib\checker-javadoc.jar;%APP_HOME%\lib\dataflow-errorprone-3.51.1-SNAPSHOT-all.jar;%APP_HOME%\lib\com.ibm.wala.cast.java.ecj-1.6.10.jar;%APP_HOME%\lib\com.ibm.wala.cast.java-1.6.12.jar;%APP_HOME%\lib\com.ibm.wala.cast-1.6.12.jar;%APP_HOME%\lib\com.ibm.wala.core-1.6.12.jar;%APP_HOME%\lib\com.ibm.wala.shrike-1.6.12.jar;%APP_HOME%\lib\com.ibm.wala.util-1.6.12.jar;%APP_HOME%\lib\org.eclipse.jdt.core-3.36.0.jar;%APP_HOME%\lib\ecj-3.37.0.jar;%APP_HOME%\lib\soot-4.4.1.jar;%APP_HOME%\lib\javax.annotation-api-1.3.2.jar;%APP_HOME%\lib\javaparser-core-3.26.2.jar;%APP_HOME%\lib\vineflower-1.10.1.jar;%APP_HOME%\lib\protobuf-java-util-3.21.2.jar;%APP_HOME%\lib\gson-2.13.1.jar;%APP_HOME%\lib\jspecify-1.0.0.jar;%APP_HOME%\lib\commons-io-2.20.0.jar;%APP_HOME%\lib\org.eclipse.core.resources-3.20.0.jar;%APP_HOME%\lib\org.eclipse.core.filesystem-1.10.200.jar;%APP_HOME%\lib\org.eclipse.text-3.13.100.jar;%APP_HOME%\lib\org.eclipse.core.expressions-3.9.200.jar;%APP_HOME%\lib\org.eclipse.core.runtime-3.30.0.jar;%APP_HOME%\lib\org.eclipse.core.jobs-3.15.100.jar;%APP_HOME%\lib\org.eclipse.core.contenttype-3.9.200.jar;%APP_HOME%\lib\org.eclipse.equinox.app-1.6.400.jar;%APP_HOME%\lib\org.eclipse.equinox.registry-3.11.400.jar;%APP_HOME%\lib\org.eclipse.equinox.preferences-3.10.400.jar;%APP_HOME%\lib\org.eclipse.core.commands-3.11.100.jar;%APP_HOME%\lib\org.eclipse.equinox.common-3.18.200.jar;%APP_HOME%\lib\dexlib2-2.5.2.jar;%APP_HOME%\lib\asm-util-9.4.jar;%APP_HOME%\lib\asm-commons-9.4.jar;%APP_HOME%\lib\asm-analysis-9.4.jar;%APP_HOME%\lib\asm-tree-9.4.jar;%APP_HOME%\lib\asm-9.4.jar;%APP_HOME%\lib\xmlpull-1.1.3.4d_b4_min.jar;%APP_HOME%\lib\axml-2.1.3.jar;%APP_HOME%\lib\polyglot-2006.jar;%APP_HOME%\lib\heros-1.2.3.jar;%APP_HOME%\lib\jasmin-3.0.3.jar;%APP_HOME%\lib\slf4j-api-2.0.3.jar;%APP_HOME%\lib\jaxb-runtime-2.4.0-b180830.0438.jar;%APP_HOME%\lib\jaxb-api-2.4.0-b180830.0359.jar;%APP_HOME%\lib\protobuf-java-3.21.7.jar;%APP_HOME%\lib\guava-31.1-android.jar;%APP_HOME%\lib\error_prone_annotations-2.38.0.jar;%APP_HOME%\lib\org.eclipse.osgi-3.18.600.jar;%APP_HOME%\lib\jsr305-3.0.2.jar;%APP_HOME%\lib\functionaljava-4.2.jar;%APP_HOME%\lib\java_cup-0.9.2.jar;%APP_HOME%\lib\javax.activation-api-1.2.0.jar;%APP_HOME%\lib\txw2-2.4.0-b180830.0438.jar;%APP_HOME%\lib\istack-commons-runtime-3.0.7.jar;%APP_HOME%\lib\stax-ex-1.8.jar;%APP_HOME%\lib\FastInfoset-1.2.15.jar;%APP_HOME%\lib\j2objc-annotations-1.3.jar;%APP_HOME%\lib\failureaccess-1.0.1.jar;%APP_HOME%\lib\listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar;%APP_HOME%\lib\checker-qual-3.12.0.jar;%APP_HOME%\lib\org.osgi.service.prefs-1.1.2.jar;%APP_HOME%\lib\osgi.annotation-8.0.1.jar


@rem Execute GenDATA
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GEN_DATA_OPTS%  -classpath "%CLASSPATH%" cfwr.CheckerFrameworkWarningResolver %*

:end
@rem End local scope for the variables with windows NT shell
if %ERRORLEVEL% equ 0 goto mainEnd

:fail
rem Set variable GEN_DATA_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% equ 0 set EXIT_CODE=1
if not ""=="%GEN_DATA_EXIT_CONSOLE%" exit %EXIT_CODE%
exit /b %EXIT_CODE%

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
