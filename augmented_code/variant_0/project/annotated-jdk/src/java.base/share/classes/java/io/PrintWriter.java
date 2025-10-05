/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.io;

    @Positive
import org.checkerframework.checker.formatter.qual.FormatMethod;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Formatter;
    @Positive
import java.util.Locale;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.IllegalCharsetNameException;
    @Positive
import java.nio.charset.UnsupportedCharsetException;

    @Positive
@AnnotatedFor({ "formatter", "index", "lock", "mustcall", "nullness" })
    @Positive
public class PrintWriter extends Writer {

    @Positive
    protected Writer out;

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter(@MustCallAlias Writer out) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter(@MustCallAlias Writer out, boolean autoFlush) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter(@MustCallAlias OutputStream out) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter(@MustCallAlias OutputStream out, boolean autoFlush) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter(@MustCallAlias OutputStream out, boolean autoFlush, Charset charset) {
    @Positive
    }

    @Positive
    public PrintWriter(String fileName) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public PrintWriter(String fileName, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public PrintWriter(String fileName, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public PrintWriter(File file) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public PrintWriter(File file, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public PrintWriter(File file, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public void flush(@GuardSatisfied PrintWriter this);

    @Positive
    public void close(@GuardSatisfied PrintWriter this);

    @Positive
    public boolean checkError(@GuardSatisfied PrintWriter this);

    @Positive
    protected void setError();

    @Positive
    protected void clearError();

    @Positive
    public void write(@GuardSatisfied PrintWriter this, int c);

    @Positive
    public void write(@GuardSatisfied PrintWriter this, char[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    public void write(@GuardSatisfied PrintWriter this, char[] buf);

    @Positive
    public void write(@GuardSatisfied PrintWriter this, String s, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    public void write(@GuardSatisfied PrintWriter this, String s);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, boolean b);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, char c);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, int i);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, long l);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, float f);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, double d);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, char[] s);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, @Nullable String s);

    @Positive
    public void print(@GuardSatisfied PrintWriter this, @Nullable Object obj);

    @Positive
    public void println(@GuardSatisfied PrintWriter this);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, boolean x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, char x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, int x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, long x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, float x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, double x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, char[] x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, @Nullable String x);

    @Positive
    public void println(@GuardSatisfied PrintWriter this, @Nullable Object x);

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter printf(@GuardSatisfied @MustCallAlias PrintWriter this, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter printf(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable Locale l, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter format(@GuardSatisfied @MustCallAlias PrintWriter this, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter format(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable Locale l, String format, @Nullable Object... args);

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable CharSequence csq);

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Positive
    @MustCallAlias
    @Positive
    public PrintWriter append(@GuardSatisfied @MustCallAlias PrintWriter this, char c);
    @Positive
}
