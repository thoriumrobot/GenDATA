/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Objects;

    @Positive
@AnnotatedFor({ "index", "interning", "mustcall", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Writer implements Appendable, Closeable, Flushable {

    @Positive
    public static Writer nullWriter();

    @Positive
    protected Object lock;

    @Positive
    protected Writer() {
    @Positive
    }

    @Positive
    protected Writer(Object lock) {
    @Positive
    }

    @Positive
    public void write(int c) throws IOException;

    @Positive
    public void write(char[] cbuf) throws IOException;

    @Positive
    public abstract void write(char[] cbuf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    public void write(String str) throws IOException;

    @Positive
    public void write(String str, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public Writer append(@MustCallAlias Writer this, @Nullable CharSequence csq) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public Writer append(@MustCallAlias Writer this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public Writer append(@MustCallAlias Writer this, char c) throws IOException;

    @Positive
    public abstract void flush() throws IOException;

    @Positive
    public abstract void close() throws IOException;
    @Positive
}
