/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.formatter.qual.FormatMethod;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Objects;
import java.util.Formatter;
import java.util.Locale;
import java.nio.charset.Charset;
import java.nio.charset.IllegalCharsetNameException;
import java.nio.charset.UnsupportedCharsetException;

@AnnotatedFor({ "formatter", "index", "lock", "mustcall", "nullness" })
public class PrintWriter extends Writer {

    protected Writer out;

    @MustCallAlias
    public PrintWriter(@MustCallAlias Writer out) {
    }

    @MustCallAlias
    public PrintWriter(@MustCallAlias Writer out, boolean autoFlush) {
    }

    @MustCallAlias
    public PrintWriter(@MustCallAlias OutputStream out) {
    }

    @MustCallAlias
    public PrintWriter(@MustCallAlias OutputStream out, boolean autoFlush) {
    }

    @MustCallAlias
    public PrintWriter(@MustCallAlias OutputStream out, boolean autoFlush, Charset charset) {
    }

    public PrintWriter(String fileName) throws FileNotFoundException {
    }

    public PrintWriter(String fileName, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    }

    public PrintWriter(String fileName, Charset charset) throws IOException {
    }

    public PrintWriter(File file) throws FileNotFoundException {
    }

    public PrintWriter(File file, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    }

    public PrintWriter(File file, Charset charset) throws IOException {
    }

    public void flush(@GuardSatisfied PrintWriter this);

    public void close(@GuardSatisfied PrintWriter this);

    public boolean checkError(@GuardSatisfied PrintWriter this);

    protected void setError();

    protected void clearError();

    public void write(@GuardSatisfied PrintWriter this, int c);

    public void write(@GuardSatisfied PrintWriter this, char[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    public void write(@GuardSatisfied PrintWriter this, char[] buf);

    public void write(@GuardSatisfied PrintWriter this, String s, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    public void write(@GuardSatisfied PrintWriter this, String s);

    public void print(@GuardSatisfied PrintWriter this, boolean b);

    public void print(@GuardSatisfied PrintWriter this, char c);

    public void print(@GuardSatisfied PrintWriter this, int i);

    public void print(@GuardSatisfied PrintWriter this, long l);

    public void print(@GuardSatisfied PrintWriter this, float f);

    public void print(@GuardSatisfied PrintWriter this, double d);

    public void print(@GuardSatisfied PrintWriter this, char[] s);

    public void print(@GuardSatisfied PrintWriter this, @Nullable String s);

    public void print(@GuardSatisfied PrintWriter this, @Nullable Object obj);

    public void println(@GuardSatisfied PrintWriter this);

    public void println(@GuardSatisfied PrintWriter this, boolean x);

    public void println(@GuardSatisfied PrintWriter this, char x);

    public void println(@GuardSatisfied PrintWriter this, int x);

    public void println(@GuardSatisfied PrintWriter this, long x);

    public void println(@GuardSatisfied PrintWriter this, float x);

    public void println(@GuardSatisfied PrintWriter this, double x);

    public void println(@GuardSatisfied PrintWriter this, char[] x);

    public void println(@GuardSatisfied PrintWriter this, @Nullable String x);

    public void println(@GuardSatisfied PrintWriter this, @Nullable Object x);

    @FormatMethod
    @MustCallAlias
    public PrintWriter printf(@GuardSatisfied @MustCallAlias PrintWriter this, String format, @Nullable Object... args);

    @FormatMethod
    @MustCallAlias
    public PrintWriter printf(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable Locale l, String format, @Nullable Object... args);

    @FormatMethod
    @MustCallAlias
    public PrintWriter format(@GuardSatisfied @MustCallAlias PrintWriter this, String format, @Nullable Object... args);

    @FormatMethod
    @MustCallAlias
    public PrintWriter format(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable Locale l, String format, @Nullable Object... args);

    @MustCallAlias
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable CharSequence csq);

    @MustCallAlias
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @MustCallAlias
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, char c);
}
