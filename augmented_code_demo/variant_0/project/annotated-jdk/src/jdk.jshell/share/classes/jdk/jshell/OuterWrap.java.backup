/*
 * Copyright (c) 2015, 2016, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jshell;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Locale;
import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;
import static jdk.jshell.Util.PARSED_LOCALE;
import static jdk.jshell.Util.REPL_CLASS_PREFIX;
import static jdk.jshell.Util.REPL_DOESNOTMATTER_CLASS_NAME;
import static jdk.jshell.Util.REPL_PACKAGE;
import static jdk.jshell.Util.expunge;

class OuterWrap implements GeneralWrap {

    protected final Wrap w;

    @Override
    public final String wrapped();

    @Override
    public int snippetIndexToWrapIndex(int ui);

    @Override
    public int wrapIndexToSnippetIndex(int si);

    @Override
    public int firstSnippetIndex();

    @Override
    public int lastSnippetIndex();

    @Override
    public int snippetLineToWrapLine(int snline);

    @Override
    public int wrapLineToSnippetLine(int wline);

    @Override
    public int firstSnippetLine();

    @Override
    public int lastSnippetLine();

    public String className();

    public String classFullName();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public String toString();

    Diag wrapDiag(Diagnostic<? extends JavaFileObject> d);

    class WrappedDiagnostic extends Diag {

        @Override
        public boolean isError();

        @Override
        public long getPosition();

        @Override
        public long getStartPosition();

        @Override
        public long getEndPosition();

        @Override
        public String getCode();

        @Override
        public String getMessage(Locale locale);

        @Override
        boolean isResolutionError();

        @Override
        public String toString();
    }
}
