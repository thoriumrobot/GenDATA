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

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.CharBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.Objects;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public abstract class Reader implements Readable, Closeable {

    public static Reader nullReader();

    protected Object lock;

    protected Reader() {
    }

    protected Reader(Object lock) {
    }

    @GTENegativeOne
    public int read(@GuardSatisfied Reader this, CharBuffer target) throws IOException;

    @GTENegativeOne
    public int read(@GuardSatisfied Reader this) throws IOException;

    @GTENegativeOne
    @LTEqLengthOf({ "#1" })
    public int read(@GuardSatisfied Reader this, char[] cbuf) throws IOException;

    @GTENegativeOne
    @LTEqLengthOf({ "#1" })
    public abstract int read(@GuardSatisfied Reader this, char[] cbuf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @NonNegative
    public long skip(@GuardSatisfied Reader this, @NonNegative long n) throws IOException;

    public boolean ready() throws IOException;

    public boolean markSupported();

    public void mark(@GuardSatisfied Reader this, @NonNegative int readAheadLimit) throws IOException;

    public void reset(@GuardSatisfied Reader this) throws IOException;

    public abstract void close(@GuardSatisfied Reader this) throws IOException;

    public long transferTo(Writer out) throws IOException;
}
