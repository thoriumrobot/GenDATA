/*
 * Copyright (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package javax.tools;

import org.checkerframework.checker.nullness.qual.Nullable;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.Locale;
import java.util.concurrent.Callable;
import javax.annotation.processing.Processor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor("nullness")
public interface JavaCompiler extends Tool, OptionChecker {

    CompilationTask getTask(@Nullable Writer out, @Nullable JavaFileManager fileManager, @Nullable DiagnosticListener<? super JavaFileObject> diagnosticListener, @Nullable Iterable<String> options, @Nullable Iterable<String> classes, @Nullable Iterable<? extends JavaFileObject> compilationUnits);

    StandardJavaFileManager getStandardFileManager(@Nullable DiagnosticListener<? super JavaFileObject> diagnosticListener, @Nullable Locale locale, @Nullable Charset charset);

    interface CompilationTask extends Callable<Boolean> {

        void addModules(Iterable<String> moduleNames);

        void setProcessors(Iterable<? extends Processor> processors);

        void setLocale(@Nullable Locale locale);

        @Override
        Boolean call();
    }
}
