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
import org.checkerframework.checker.i18n.qual.Localized;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.mustcall.qual.NotOwning;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.util.Formatter;
import java.util.Locale;
import java.nio.charset.Charset;
import java.nio.charset.IllegalCharsetNameException;
import java.nio.charset.UnsupportedCharsetException;

@CFComment({ "lock: TODO: Should parameters be @GuardSatisfied, or is the default of @GuardedBy({}) appropriate? (@GuardedBy({}) is more conservative.)" })
@AnnotatedFor({ "formatter", "i18n", "index", "lock", "mustcall", "nullness", "signedness" })
public class PrintStream extends FilterOutputStream implements Appendable, Closeable {

    @MustCallAlias
    public PrintStream(@MustCallAlias OutputStream out) {
    }

    @MustCallAlias
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush) {
    }

    @MustCallAlias
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush, String encoding) throws UnsupportedEncodingException {
    }

    @MustCallAlias
    public PrintStream(@MustCallAlias OutputStream out, boolean autoFlush, Charset charset) {
    }

    public PrintStream(String fileName) throws FileNotFoundException {
    }

    public PrintStream(String fileName, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    }

    public PrintStream(String fileName, Charset charset) throws IOException {
    }

    public PrintStream(File file) throws FileNotFoundException {
    }

    public PrintStream(File file, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    }

    public PrintStream(File file, Charset charset) throws IOException {
    }

    @Override
    public void flush(@GuardSatisfied PrintStream this);

    @Override
    public void close(@GuardSatisfied PrintStream this);

    public boolean checkError(@GuardSatisfied PrintStream this);

    protected void setError();

    protected void clearError();

    @Override
    public void write(@GuardSatisfied PrintStream this, int b);

    @Override
    public void write(@GuardSatisfied PrintStream this, @PolySigned byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len);

    @Override
    public void write(@GuardSatisfied PrintStream this, @PolySigned byte[] buf) throws IOException;

    public void writeBytes(@GuardSatisfied PrintStream this, @PolySigned byte[] buf);

    public void print(@GuardSatisfied PrintStream this, boolean b);

    public void print(@GuardSatisfied PrintStream this, char c);

    public void print(@GuardSatisfied PrintStream this, int i);

    public void print(@GuardSatisfied PrintStream this, long l);

    public void print(@GuardSatisfied PrintStream this, float f);

    public void print(@GuardSatisfied PrintStream this, double d);

    public void print(@GuardSatisfied PrintStream this, @PolySigned char[] s);

    public void print(@GuardSatisfied PrintStream this, @Nullable String s);

    public void print(@GuardSatisfied PrintStream this, @Nullable Object obj);

    public void println(@GuardSatisfied PrintStream this);

    public void println(@GuardSatisfied PrintStream this, boolean x);

    public void println(@GuardSatisfied PrintStream this, char x);

    public void println(@GuardSatisfied PrintStream this, int x);

    public void println(@GuardSatisfied PrintStream this, long x);

    public void println(@GuardSatisfied PrintStream this, float x);

    public void println(@GuardSatisfied PrintStream this, double x);

    public void println(@GuardSatisfied PrintStream this, char[] x);

    public void println(@GuardSatisfied PrintStream this, @Nullable @Localized String x);

    public void println(@GuardSatisfied PrintStream this, @Nullable Object x);

    @CFComment({ "lock/nullness: The vararg arrays can actually be null, but let's not annotate them because passing null is bad style; see whether this annotation is useful." })
    @FormatMethod
    @NotOwning
    public PrintStream printf(@GuardSatisfied PrintStream this, String format, @Nullable Object... args);

    @FormatMethod
    @NotOwning
    public PrintStream printf(@GuardSatisfied PrintStream this, @Nullable Locale l, String format, @Nullable Object... args);

    @FormatMethod
    @NotOwning
    public PrintStream format(@GuardSatisfied PrintStream this, String format, @Nullable Object... args);

    @FormatMethod
    @NotOwning
    public PrintStream format(@GuardSatisfied PrintStream this, @Nullable Locale l, String format, @Nullable Object... args);

    @MustCallAlias
    public PrintStream append(@MustCallAlias PrintStream this, @Nullable CharSequence csq);

    @MustCallAlias
    public PrintStream append(@MustCallAlias PrintStream this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end);

    @MustCallAlias
    public PrintStream append(@MustCallAlias PrintStream this, char c);
}
