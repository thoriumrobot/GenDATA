/*
 * Copyright (c) 1996, 2018, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Objects;

@AnnotatedFor({ "index", "interning", "mustcall", "nullness" })
@UsesObjectEquals
public abstract class Writer implements Appendable, Closeable, Flushable {

    public static Writer nullWriter();

    protected Object lock;

    protected Writer() {
    }

    protected Writer(Object lock) {
    }

    public void write(int c) throws IOException;

    public void write(char[] cbuf) throws IOException;

    public abstract void write(char[] cbuf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    public void write(String str) throws IOException;

    public void write(String str, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @MustCallAlias
    public Writer append(@MustCallAlias Writer this, @Nullable CharSequence csq) throws IOException;

    @MustCallAlias
    public Writer append(@MustCallAlias Writer this, @Nullable CharSequence csq, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int end) throws IOException;

    @MustCallAlias
    public Writer append(@MustCallAlias Writer this, char c) throws IOException;

    public abstract void flush() throws IOException;

    public abstract void close() throws IOException;
}
