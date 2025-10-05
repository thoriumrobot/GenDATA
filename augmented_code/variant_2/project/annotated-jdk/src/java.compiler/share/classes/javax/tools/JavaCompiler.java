/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package javax.tools;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.io.Writer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.util.Locale;
    @Positive
import java.util.concurrent.Callable;
    @Positive
import javax.annotation.processing.Processor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public interface JavaCompiler extends Tool, OptionChecker {

    @Positive
    CompilationTask getTask(@Nullable Writer out, @Nullable JavaFileManager fileManager, @Nullable DiagnosticListener<? super JavaFileObject> diagnosticListener, @Nullable Iterable<String> options, @Nullable Iterable<String> classes, @Nullable Iterable<? extends JavaFileObject> compilationUnits);

    @Positive
    StandardJavaFileManager getStandardFileManager(@Nullable DiagnosticListener<? super JavaFileObject> diagnosticListener, @Nullable Locale locale, @Nullable Charset charset);

    @Positive
    interface CompilationTask extends Callable<Boolean> {

    @Positive
        void addModules(Iterable<String> moduleNames);

    @Positive
        void setProcessors(Iterable<? extends Processor> processors);

    @Positive
        void setLocale(@Nullable Locale locale);

    @Positive
        @Override
    @Positive
        Boolean call();
    @Positive
    }
    @Positive
}
