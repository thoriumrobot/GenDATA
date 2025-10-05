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
import org.checkerframework.checker.i18n.qual.Localized;
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
import org.checkerframework.checker.mustcall.qual.NotOwning;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
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
@CFComment({ "lock: TODO: Should parameters be @GuardSatisfied, or is the default of @GuardedBy({}) appropriate? (@GuardedBy({}) is more conservative.)" })
    @Positive
@AnnotatedFor({ "formatter", "i18n", "index", "lock", "mustcall", "nullness", "signedness" })
    @Positive
public class PrintStream extends FilterOutputStream implements Appendable, Closeable {

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream(@MustCallAlias OutputStream out) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush, String encoding) throws UnsupportedEncodingException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush, Charset charset) {
    @Positive
    }

    @Positive
    public PrintStream(String fileName) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public PrintStream(String fileName, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public PrintStream(String fileName, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public PrintStream(File file) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public PrintStream(File file, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public PrintStream(File file, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void flush(@GuardSatisfied PrintStream this);

    @Positive
    @Override
    @Positive
    public void close(@GuardSatisfied PrintStream this);

    @Positive
    public boolean checkError(@GuardSatisfied PrintStream this);

    @Positive
    protected void setError();

    @Positive
    protected void clearError();

    @Positive
    @Override
    @Positive
    public void write(@GuardSatisfied PrintStream this, int b);

    @Positive
    @Override
    @Positive
    public void write(@GuardSatisfied PrintStream this, @PolySigned byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Positive
    @Override
    @Positive
    public void write(@GuardSatisfied PrintStream this, @PolySigned byte[] buf) throws IOException;

    @Positive
    public void writeBytes(@GuardSatisfied PrintStream this, @PolySigned byte[] buf);

    @Positive
    public void print(@GuardSatisfied PrintStream this, boolean b);

    @Positive
    public void print(@GuardSatisfied PrintStream this, char c);

    @Positive
    public void print(@GuardSatisfied PrintStream this, int i);

    @Positive
    public void print(@GuardSatisfied PrintStream this, long l);

    @Positive
    public void print(@GuardSatisfied PrintStream this, float f);

    @Positive
    public void print(@GuardSatisfied PrintStream this, double d);

    @Positive
    public void print(@GuardSatisfied PrintStream this, @PolySigned char[] s);

    @Positive
    public void print(@GuardSatisfied PrintStream this, @Nullable String s);

    @Positive
    public void print(@GuardSatisfied PrintStream this, @Nullable Object obj);

    @Positive
    public void println(@GuardSatisfied PrintStream this);

    @Positive
    public void println(@GuardSatisfied PrintStream this, boolean x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, char x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, int x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, long x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, float x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, double x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, char[] x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, @Nullable @Localized String x);

    @Positive
    public void println(@GuardSatisfied PrintStream this, @Nullable Object x);

    @Positive
    @CFComment({ "lock/nullness: The vararg arrays can actually be null, but let's not annotate them because passing null is bad style; see whether this annotation is useful." })
    @Positive
    @FormatMethod
    @Positive
    @NotOwning
    @Positive
    public PrintStream printf(@GuardSatisfied PrintStream this, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @NotOwning
    @Positive
    public PrintStream printf(@GuardSatisfied PrintStream this, @Nullable Locale l, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @NotOwning
    @Positive
    public PrintStream format(@GuardSatisfied PrintStream this, String format, @Nullable Object... args);

    @Positive
    @FormatMethod
    @Positive
    @NotOwning
    @Positive
    public PrintStream format(@GuardSatisfied PrintStream this, @Nullable Locale l, String format, @Nullable Object... args);

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream append(@MustCallAlias PrintStream this, @Nullable CharSequence csq);

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream append(@MustCallAlias PrintStream this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @Positive
    @MustCallAlias
    @Positive
    public PrintStream append(@MustCallAlias PrintStream this, char c);
    @Positive
}

// CFWR semantic augmentation - variant 0
